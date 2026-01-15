import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CV Edge App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  final List<Widget> _pages = const [
    SegmentationPage(),
    OcrPage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.view_in_ar),
            selectedIcon: Icon(Icons.view_in_ar),
            label: 'Segmentation',
          ),
          NavigationDestination(
            icon: Icon(Icons.document_scanner_outlined),
            selectedIcon: Icon(Icons.document_scanner),
            label: 'OCR',
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// Instance Segmentation Page
// ============================================================================

class SegmentationPage extends StatefulWidget {
  const SegmentationPage({super.key});

  @override
  State<SegmentationPage> createState() => _SegmentationPageState();
}

class _SegmentationPageState extends State<SegmentationPage> {
  File? _selectedImage;
  ui.Image? _decodedImage;
  List<YOLOResult> _detectionResults = [];
  bool _isProcessing = false;
  bool _isModelLoaded = false;
  final ImagePicker _picker = ImagePicker();
  YOLO? _yolo;
  int? _selectedResultIndex;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _yolo = YOLO(
        modelPath: 'yolo11n.tflite',
        task: YOLOTask.detect,
      );
      await _yolo!.loadModel();
      setState(() {
        _isModelLoaded = true;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading model: $e')),
        );
      }
    }
  }

  @override
  void dispose() {
    _yolo?.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      final file = File(image.path);
      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      setState(() {
        _selectedImage = file;
        _decodedImage = frame.image;
        _detectionResults = [];
        _selectedResultIndex = null;
      });
    }
  }

  Future<void> _performDetection() async {
    if (_selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select an image first')),
      );
      return;
    }

    if (!_isModelLoaded || _yolo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model is still loading...')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _detectionResults = [];
      _selectedResultIndex = null;
    });

    try {
      final imageBytes = await _selectedImage!.readAsBytes();
      final resultMap = await _yolo!.predict(imageBytes);
      debugPrint('YOLO inference completed. Raw result keys: ${resultMap.keys.toList()}');
      debugPrint('Number of boxes: ${(resultMap['boxes'] as List?)?.length ?? 0}');

      // Debug: print raw box data to understand the format
      final rawBoxes = resultMap['boxes'] as List<dynamic>? ?? [];
      if (rawBoxes.isNotEmpty) {
        debugPrint('Raw box[0] keys: ${(rawBoxes[0] as Map).keys.toList()}');
        debugPrint('Raw box[0] data: ${rawBoxes[0]}');
      }

      // Also check detections format
      final rawDetections = resultMap['detections'] as List<dynamic>? ?? [];
      if (rawDetections.isNotEmpty) {
        debugPrint('Raw detection[0]: ${rawDetections[0]}');
      }

      // Parse results manually since YOLOResult.fromMap doesn't work correctly
      final List<YOLOResult> results = [];
      for (final box in rawBoxes) {
        final map = box as Map<String, dynamic>;

        // Extract bounding box - try different possible key names
        double left = 0, top = 0, right = 0, bottom = 0;

        if (map.containsKey('x1')) {
          left = (map['x1'] as num?)?.toDouble() ?? 0;
          top = (map['y1'] as num?)?.toDouble() ?? 0;
          right = (map['x2'] as num?)?.toDouble() ?? 0;
          bottom = (map['y2'] as num?)?.toDouble() ?? 0;
        } else if (map.containsKey('rect')) {
          final rect = map['rect'];
          if (rect is Map) {
            left = (rect['left'] as num?)?.toDouble() ?? 0;
            top = (rect['top'] as num?)?.toDouble() ?? 0;
            right = (rect['right'] as num?)?.toDouble() ?? 0;
            bottom = (rect['bottom'] as num?)?.toDouble() ?? 0;
          }
        } else if (map.containsKey('boundingBox')) {
          final bbox = map['boundingBox'];
          if (bbox is Map) {
            left = (bbox['left'] as num?)?.toDouble() ?? 0;
            top = (bbox['top'] as num?)?.toDouble() ?? 0;
            right = (bbox['right'] as num?)?.toDouble() ?? 0;
            bottom = (bbox['bottom'] as num?)?.toDouble() ?? 0;
          }
        }

        // Extract confidence - normalize if > 1
        double confidence = (map['confidence'] as num?)?.toDouble() ?? 0;
        if (confidence > 1) {
          confidence = confidence / 100.0;
        }

        // Extract class info
        final className = map['className'] as String? ??
                         map['label'] as String? ??
                         map['class'] as String? ??
                         'Unknown';
        final classIndex = (map['classIndex'] as num?)?.toInt() ??
                          (map['index'] as num?)?.toInt() ?? 0;

        final boundingBox = Rect.fromLTRB(left, top, right, bottom);

        debugPrint('Parsed: $className ($confidence), bbox: $boundingBox');

        // Only add if we have a valid bounding box
        if (boundingBox.width > 0 && boundingBox.height > 0) {
          results.add(YOLOResult(
            classIndex: classIndex,
            className: className,
            confidence: confidence,
            boundingBox: boundingBox,
            normalizedBox: Rect.zero,
          ));
        }
      }

      // Filter by confidence threshold (50%) and sort by confidence (descending)
      const double confidenceThreshold = 0.5;
      final filteredResults = results.where((r) => r.confidence >= confidenceThreshold).toList();
      filteredResults.sort((a, b) => b.confidence.compareTo(a.confidence));

      debugPrint('Found ${filteredResults.length} results above ${(confidenceThreshold * 100).toInt()}% threshold');

      setState(() {
        _detectionResults = filteredResults;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error during detection: $e')),
        );
      }
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Instance Segmentation'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Model status indicator
            if (!_isModelLoaded)
              Container(
                padding: const EdgeInsets.all(12),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.orange[100],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Row(
                  children: [
                    SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    SizedBox(width: 12),
                    Text('Loading YOLO26 model...'),
                  ],
                ),
              ),

            // Image display area with bounding boxes
            Container(
              height: 350,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: _selectedImage != null && _decodedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: CustomPaint(
                        painter: DetectionPainter(
                          image: _decodedImage!,
                          results: _detectionResults,
                          selectedIndex: _selectedResultIndex,
                        ),
                        size: Size.infinite,
                      ),
                    )
                  : const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.image, size: 64, color: Colors.grey),
                          SizedBox(height: 8),
                          Text('No image selected',
                              style: TextStyle(color: Colors.grey)),
                        ],
                      ),
                    ),
            ),
            const SizedBox(height: 16),

            // Action buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickImage,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Select Image'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed:
                        _isProcessing || !_isModelLoaded ? null : _performDetection,
                    icon: _isProcessing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.search),
                    label: Text(_isProcessing ? 'Processing...' : 'Detect'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Results section
            if (_detectionResults.isNotEmpty) ...[
              Row(
                children: [
                  Text(
                    'Top ${_detectionResults.length} detection(s):',
                    style:
                        const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const Spacer(),
                  if (_selectedResultIndex != null)
                    TextButton(
                      onPressed: () {
                        setState(() {
                          _selectedResultIndex = null;
                        });
                      },
                      child: const Text('Show all'),
                    ),
                ],
              ),
              const SizedBox(height: 8),
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _detectionResults.length,
                itemBuilder: (context, index) {
                  final result = _detectionResults[index];
                  final isSelected = _selectedResultIndex == index;
                  // Clamp confidence to 0-1 range for display
                  final displayConfidence = result.confidence.clamp(0.0, 1.0);
                  return Card(
                    margin: const EdgeInsets.only(bottom: 8),
                    color: isSelected
                        ? _getColorForIndex(index).withOpacity(0.15)
                        : null,
                    shape: isSelected
                        ? RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                            side: BorderSide(
                              color: _getColorForIndex(index),
                              width: 2,
                            ),
                          )
                        : null,
                    child: ListTile(
                      onTap: () {
                        setState(() {
                          _selectedResultIndex = isSelected ? null : index;
                        });
                      },
                      leading: CircleAvatar(
                        backgroundColor: _getColorForIndex(index),
                        child: Text(
                          '${index + 1}',
                          style: const TextStyle(color: Colors.white),
                        ),
                      ),
                      title: Text(result.className),
                      subtitle: Text(
                        'Confidence: ${(displayConfidence * 100).toStringAsFixed(1)}%',
                      ),
                      trailing: Icon(
                        isSelected ? Icons.visibility : Icons.visibility_outlined,
                        color: isSelected ? _getColorForIndex(index) : Colors.grey,
                      ),
                    ),
                  );
                },
              ),
            ] else if (_selectedImage != null && !_isProcessing) ...[
              const Center(
                child: Text(
                  'Press Detect to find objects in the image',
                  style: TextStyle(color: Colors.grey),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Color _getColorForIndex(int index) {
    final colors = [
      Colors.red,
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.indigo,
      Colors.amber,
      Colors.cyan,
    ];
    return colors[index % colors.length];
  }
}

// Custom painter to draw bounding boxes on the image
class DetectionPainter extends CustomPainter {
  final ui.Image image;
  final List<YOLOResult> results;
  final int? selectedIndex;

  DetectionPainter({
    required this.image,
    required this.results,
    this.selectedIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale to fit image in canvas
    final double scaleX = size.width / image.width;
    final double scaleY = size.height / image.height;
    final double scale = scaleX < scaleY ? scaleX : scaleY;

    final double offsetX = (size.width - image.width * scale) / 2;
    final double offsetY = (size.height - image.height * scale) / 2;

    // Draw the image
    canvas.save();
    canvas.translate(offsetX, offsetY);
    canvas.scale(scale);
    canvas.drawImage(image, Offset.zero, Paint());

    // Draw bounding boxes (only selected one if there's a selection)
    for (int i = 0; i < results.length; i++) {
      // Skip non-selected items if one is selected
      if (selectedIndex != null && selectedIndex != i) {
        continue;
      }

      final result = results[i];
      final color = _getColorForIndex(i);

      // Try to determine the correct rect to use
      // If boundingBox values are small (0-1 range), it's likely normalized
      // If normalizedBox is available and boundingBox seems wrong, use normalizedBox
      Rect rect = result.boundingBox;
      final normalizedRect = result.normalizedBox;

      // Check if boundingBox is in normalized coords (all values between 0 and 1)
      final isBboxNormalized = rect.left >= 0 &&
          rect.left <= 1 &&
          rect.top >= 0 &&
          rect.top <= 1 &&
          rect.right >= 0 &&
          rect.right <= 1 &&
          rect.bottom >= 0 &&
          rect.bottom <= 1;

      if (isBboxNormalized) {
        // Convert normalized to pixel coordinates
        rect = Rect.fromLTRB(
          rect.left * image.width,
          rect.top * image.height,
          rect.right * image.width,
          rect.bottom * image.height,
        );
      } else if (rect.width <= 0 || rect.height <= 0) {
        // boundingBox is invalid, try normalizedBox
        rect = Rect.fromLTRB(
          normalizedRect.left * image.width,
          normalizedRect.top * image.height,
          normalizedRect.right * image.width,
          normalizedRect.bottom * image.height,
        );
      }

      // Draw bounding box
      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0 / scale;

      canvas.drawRect(rect, boxPaint);

      // Draw label background
      final labelBgPaint = Paint()
        ..color = color
        ..style = PaintingStyle.fill;

      // Clamp confidence to 0-1 range for display
      final displayConfidence = result.confidence.clamp(0.0, 1.0);
      final labelText =
          '${result.className} ${(displayConfidence * 100).toStringAsFixed(0)}%';
      final textSpan = TextSpan(
        text: labelText,
        style: TextStyle(
          color: Colors.white,
          fontSize: 14.0 / scale,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final labelRect = Rect.fromLTWH(
        rect.left,
        rect.top - textPainter.height - 4 / scale,
        textPainter.width + 8 / scale,
        textPainter.height + 4 / scale,
      );
      canvas.drawRect(labelRect, labelBgPaint);

      // Draw label text
      textPainter.paint(
        canvas,
        Offset(rect.left + 4 / scale, rect.top - textPainter.height - 2 / scale),
      );
    }

    canvas.restore();
  }

  Color _getColorForIndex(int index) {
    final colors = [
      Colors.red,
      Colors.blue,
      Colors.green,
      Colors.orange,
      Colors.purple,
      Colors.teal,
      Colors.pink,
      Colors.indigo,
      Colors.amber,
      Colors.cyan,
    ];
    return colors[index % colors.length];
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.image != image ||
        oldDelegate.results != results ||
        oldDelegate.selectedIndex != selectedIndex;
  }
}

// ============================================================================
// OCR Page (existing functionality)
// ============================================================================

class OcrPage extends StatefulWidget {
  const OcrPage({super.key});

  @override
  State<OcrPage> createState() => _OcrPageState();
}

class _OcrPageState extends State<OcrPage> {
  File? _selectedImage;
  String _recognizedText = '';
  List<TextBlock> _textBlocks = [];
  bool _isProcessing = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _recognizedText = '';
        _textBlocks = [];
      });
    }
  }

  Future<void> _performOcr() async {
    if (_selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select an image first')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _recognizedText = '';
      _textBlocks = [];
    });

    try {
      final inputImage = InputImage.fromFile(_selectedImage!);
      final textRecognizer =
          TextRecognizer(script: TextRecognitionScript.latin);
      final RecognizedText recognizedText =
          await textRecognizer.processImage(inputImage);

      setState(() {
        _recognizedText = recognizedText.text;
        _textBlocks = recognizedText.blocks;
      });

      await textRecognizer.close();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error during OCR: $e')),
        );
      }
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('OCR Text Recognition'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image display area
            Container(
              height: 250,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: _selectedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        _selectedImage!,
                        fit: BoxFit.contain,
                      ),
                    )
                  : const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.image, size: 64, color: Colors.grey),
                          SizedBox(height: 8),
                          Text('No image selected',
                              style: TextStyle(color: Colors.grey)),
                        ],
                      ),
                    ),
            ),
            const SizedBox(height: 16),

            // Action buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickImage,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Select Image'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isProcessing ? null : _performOcr,
                    icon: _isProcessing
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.document_scanner),
                    label: Text(_isProcessing ? 'Processing...' : 'OCR'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Results section
            if (_recognizedText.isNotEmpty) ...[
              const Text(
                'Recognized Text:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: SelectableText(
                  _recognizedText,
                  style: const TextStyle(fontSize: 16),
                ),
              ),
              const SizedBox(height: 24),

              // Text blocks with location info
              const Text(
                'Text Blocks with Location:',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _textBlocks.length,
                itemBuilder: (context, index) {
                  final block = _textBlocks[index];
                  final rect = block.boundingBox;
                  return Card(
                    margin: const EdgeInsets.only(bottom: 8),
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Block ${index + 1}',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.deepPurple,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            block.text,
                            style: const TextStyle(fontSize: 14),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Location: (${rect.left.toInt()}, ${rect.top.toInt()}) - '
                            '(${rect.right.toInt()}, ${rect.bottom.toInt()})',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                          Text(
                            'Size: ${rect.width.toInt()} x ${rect.height.toInt()} px',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ] else if (_selectedImage != null && !_isProcessing) ...[
              const Center(
                child: Text(
                  'Press OCR to recognize text in the image',
                  style: TextStyle(color: Colors.grey),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
