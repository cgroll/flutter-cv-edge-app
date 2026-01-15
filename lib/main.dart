import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

List<CameraDescription> cameras = [];

// Global YOLO model instance for preloading
YOLO? globalYoloModel;
bool isGlobalModelLoaded = false;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize cameras
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Error initializing cameras: $e');
  }

  // Preload YOLO model in background
  _preloadYoloModel();

  runApp(const MyApp());
}

Future<void> _preloadYoloModel() async {
  try {
    globalYoloModel = YOLO(
      modelPath: 'yolo11n.tflite',
      task: YOLOTask.detect,
    );
    await globalYoloModel!.loadModel();
    isGlobalModelLoaded = true;
    debugPrint('YOLO model preloaded successfully');
  } catch (e) {
    debugPrint('Error preloading YOLO model: $e');
  }
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
    CameraDetectionPage(),
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
            label: 'Detection',
          ),
          NavigationDestination(
            icon: Icon(Icons.videocam_outlined),
            selectedIcon: Icon(Icons.videocam),
            label: 'Live',
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
// Object Detection Page
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
  final ImagePicker _picker = ImagePicker();
  int? _selectedResultIndex;
  double? _lastInferenceTimeSeconds;

  // Use the preloaded global model
  bool get _isModelLoaded => isGlobalModelLoaded;
  YOLO? get _yolo => globalYoloModel;

  @override
  void initState() {
    super.initState();
    // Check if model is still loading
    if (!isGlobalModelLoaded) {
      _waitForModel();
    }
  }

  Future<void> _waitForModel() async {
    // Poll until model is loaded
    while (!isGlobalModelLoaded && mounted) {
      await Future.delayed(const Duration(milliseconds: 100));
      if (mounted) setState(() {});
    }
  }

  @override
  void dispose() {
    // Model is global, don't dispose here
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
      final stopwatch = Stopwatch()..start();
      final resultMap = await _yolo!.predict(imageBytes);
      stopwatch.stop();
      final inferenceSeconds = stopwatch.elapsedMilliseconds / 1000.0;
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
        _lastInferenceTimeSeconds = inferenceSeconds;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Detection completed in ${inferenceSeconds.toStringAsFixed(1)}s'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
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
        title: const Text('Object Detection'),
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
// Camera Live Detection Page
// ============================================================================

class CameraDetectionPage extends StatefulWidget {
  const CameraDetectionPage({super.key});

  @override
  State<CameraDetectionPage> createState() => _CameraDetectionPageState();
}

class _CameraDetectionPageState extends State<CameraDetectionPage> {
  CameraController? _cameraController;
  bool _isProcessing = false;
  bool _isStreaming = false;
  List<YOLOResult> _detectionResults = [];
  ui.Image? _displayImage;
  DateTime? _lastProcessedTime;
  double _currentFps = 0;
  CameraLensDirection _currentLensDirection = CameraLensDirection.back;
  int _sensorOrientation = 0;

  // Target 30 FPS = 33ms between frames
  static const int _targetFrameIntervalMs = 33;

  // Detection statistics
  int _totalFramesProcessed = 0;
  final Map<String, int> _classDetectionCounts = {};
  final Map<String, List<double>> _classConfidences = {};
  final List<double> _inferenceTimes = [];

  bool get _isModelLoaded => isGlobalModelLoaded;
  YOLO? get _yolo => globalYoloModel;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    // Check if model is still loading
    if (!isGlobalModelLoaded) {
      _waitForModel();
    }
  }

  Future<void> _waitForModel() async {
    // Poll until model is loaded
    while (!isGlobalModelLoaded && mounted) {
      await Future.delayed(const Duration(milliseconds: 100));
      if (mounted) setState(() {});
    }
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No cameras available')),
        );
      }
      return;
    }

    // Use the camera matching current lens direction, or first available
    final camera = cameras.firstWhere(
      (c) => c.lensDirection == _currentLensDirection,
      orElse: () => cameras.first,
    );

    // Store sensor orientation for rotation correction
    _sensorOrientation = camera.sensorOrientation;
    debugPrint('Camera sensor orientation: $_sensorOrientation degrees');

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();
      if (mounted) {
        setState(() {});
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing camera: $e')),
        );
      }
    }
  }

  Future<void> _switchCamera() async {
    // Stop streaming if active
    final wasStreaming = _isStreaming;
    if (wasStreaming) {
      _stopStreaming();
    }

    // Dispose current controller
    await _cameraController?.dispose();

    // Toggle lens direction
    setState(() {
      _currentLensDirection = _currentLensDirection == CameraLensDirection.back
          ? CameraLensDirection.front
          : CameraLensDirection.back;
      _displayImage = null;
      _detectionResults = [];
    });

    // Initialize with new camera
    await _initializeCamera();

    // Resume streaming if it was active
    if (wasStreaming && mounted) {
      _startStreaming();
    }
  }

  @override
  void dispose() {
    _stopStreaming();
    _cameraController?.dispose();
    // Model is global, don't dispose here
    super.dispose();
  }

  void _startStreaming() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }
    if (!_isModelLoaded || _yolo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model is still loading...')),
      );
      return;
    }

    // Reset statistics
    _totalFramesProcessed = 0;
    _classDetectionCounts.clear();
    _classConfidences.clear();
    _inferenceTimes.clear();

    setState(() {
      _isStreaming = true;
      _lastProcessedTime = DateTime.now();
    });

    _cameraController!.startImageStream(_onCameraFrame);
  }

  void _stopStreaming() {
    if (_cameraController != null && _cameraController!.value.isStreamingImages) {
      _cameraController!.stopImageStream();
    }

    // Print detection statistics
    _printDetectionStatistics();

    setState(() {
      _isStreaming = false;
      _isProcessing = false;
    });
  }

  void _printDetectionStatistics() {
    if (_totalFramesProcessed == 0) {
      debugPrint('=== Detection Statistics ===');
      debugPrint('No frames were processed.');
      return;
    }

    debugPrint('=== Detection Statistics ===');
    debugPrint('Total frames processed: $_totalFramesProcessed');
    debugPrint('');

    // Inference timing statistics
    if (_inferenceTimes.isNotEmpty) {
      final avgTime = _inferenceTimes.reduce((a, b) => a + b) / _inferenceTimes.length;
      final minTime = _inferenceTimes.reduce((a, b) => a < b ? a : b);
      final maxTime = _inferenceTimes.reduce((a, b) => a > b ? a : b);
      debugPrint('Inference timing (incl. pre/post processing):');
      debugPrint('  Average: ${avgTime.toStringAsFixed(1)} ms');
      debugPrint('  Min: ${minTime.toStringAsFixed(1)} ms');
      debugPrint('  Max: ${maxTime.toStringAsFixed(1)} ms');
      debugPrint('');
    }

    debugPrint('Class detection counts:');

    // Sort by detection count (descending)
    final sortedEntries = _classDetectionCounts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    for (final entry in sortedEntries) {
      final percentage = (entry.value / _totalFramesProcessed * 100).toStringAsFixed(1);
      final confidences = _classConfidences[entry.key] ?? [];
      final avgConfidence = confidences.isNotEmpty
          ? (confidences.reduce((a, b) => a + b) / confidences.length * 100).toStringAsFixed(1)
          : '0.0';
      debugPrint('  ${entry.key}: ${entry.value}/$_totalFramesProcessed frames ($percentage%), avg confidence: $avgConfidence%');
    }

    if (_classDetectionCounts.isEmpty) {
      debugPrint('  No objects detected in any frame.');
    }
    debugPrint('============================');
  }

  void _onCameraFrame(CameraImage cameraImage) {
    // Always update the latest frame reference (for picking up most recent)
    // We convert the CameraImage to bytes for processing
    // Only process if not already processing (congestion control)
    if (!_isProcessing && _isStreaming) {
      _processFrame(cameraImage);
    }
  }

  Future<void> _processFrame(CameraImage cameraImage) async {
    if (!mounted || !_isStreaming) return;

    // Check frame rate limiting (target 10 FPS)
    final now = DateTime.now();
    if (_lastProcessedTime != null) {
      final elapsed = now.difference(_lastProcessedTime!).inMilliseconds;
      if (elapsed < _targetFrameIntervalMs) {
        return;
      }
    }

    setState(() {
      _isProcessing = true;
    });

    try {
      final stopwatch = Stopwatch()..start();

      // Convert CameraImage to JPEG bytes
      final Uint8List? imageBytes = await _convertCameraImageToJpeg(cameraImage);
      if (imageBytes == null) {
        setState(() {
          _isProcessing = false;
        });
        return;
      }

      // Decode for display
      final codec = await ui.instantiateImageCodec(imageBytes);
      final frame = await codec.getNextFrame();

      // Run YOLO detection
      final resultMap = await _yolo!.predict(imageBytes);

      stopwatch.stop();
      final inferenceTimeMs = stopwatch.elapsedMilliseconds.toDouble();
      _inferenceTimes.add(inferenceTimeMs);

      // Parse results (same logic as SegmentationPage)
      final List<YOLOResult> results = [];
      final rawBoxes = resultMap['boxes'] as List<dynamic>? ?? [];

      for (final box in rawBoxes) {
        final map = box as Map<String, dynamic>;

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

        double confidence = (map['confidence'] as num?)?.toDouble() ?? 0;
        if (confidence > 1) {
          confidence = confidence / 100.0;
        }

        final className = map['className'] as String? ??
            map['label'] as String? ??
            map['class'] as String? ??
            'Unknown';
        final classIndex = (map['classIndex'] as num?)?.toInt() ??
            (map['index'] as num?)?.toInt() ?? 0;

        final boundingBox = Rect.fromLTRB(left, top, right, bottom);

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

      // Filter by confidence threshold (35% for live detection to reduce flicker)
      const double confidenceThreshold = 0.35;
      final filteredResults = results.where((r) => r.confidence >= confidenceThreshold).toList();
      filteredResults.sort((a, b) => b.confidence.compareTo(a.confidence));

      // Update detection statistics
      _totalFramesProcessed++;
      // Track unique classes detected in this frame (count each class only once per frame)
      final Set<String> classesInFrame = {};
      for (final result in filteredResults) {
        classesInFrame.add(result.className);
        // Track confidence for averaging
        _classConfidences.putIfAbsent(result.className, () => []);
        _classConfidences[result.className]!.add(result.confidence);
      }
      for (final className in classesInFrame) {
        _classDetectionCounts[className] = (_classDetectionCounts[className] ?? 0) + 1;
      }

      // Calculate FPS
      final totalElapsed = now.difference(_lastProcessedTime!).inMilliseconds;
      if (totalElapsed > 0) {
        _currentFps = 1000.0 / totalElapsed;
      }
      _lastProcessedTime = now;

      if (mounted) {
        setState(() {
          _displayImage = frame.image;
          _detectionResults = filteredResults;
          _isProcessing = false;
        });
      }
    } catch (e) {
      debugPrint('Error processing frame: $e');
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  Future<Uint8List?> _convertCameraImageToJpeg(CameraImage cameraImage) async {
    try {
      // For JPEG format (Android with imageFormatGroup: ImageFormatGroup.jpeg)
      if (cameraImage.format.group == ImageFormatGroup.jpeg) {
        return cameraImage.planes[0].bytes;
      }

      // For YUV420 format (common on Android), convert using image package
      if (cameraImage.format.group == ImageFormatGroup.yuv420) {
        return _convertYUV420ToJpeg(cameraImage);
      }

      // For BGRA format (iOS)
      if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
        return _convertBGRA8888ToJpeg(cameraImage);
      }

      return null;
    } catch (e) {
      debugPrint('Error converting camera image: $e');
      return null;
    }
  }

  Uint8List? _convertYUV420ToJpeg(CameraImage cameraImage) {
    try {
      final int width = cameraImage.width;
      final int height = cameraImage.height;

      final yPlane = cameraImage.planes[0];
      final uPlane = cameraImage.planes[1];
      final vPlane = cameraImage.planes[2];

      // Check pixel stride to determine format (NV21 vs YUV420)
      final int uvPixelStride = uPlane.bytesPerPixel ?? 1;
      final int uvRowStride = uPlane.bytesPerRow;

      var image = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yPlane.bytesPerRow + x;

          // For NV21/NV12 format (interleaved UV), pixel stride is 2
          // For YUV420 planar, pixel stride is 1
          final int uvX = x ~/ 2;
          final int uvY = y ~/ 2;
          final int uvIndex = uvY * uvRowStride + uvX * uvPixelStride;

          final int yValue = yPlane.bytes[yIndex];

          // Read U and V values
          int uValue, vValue;
          if (uvPixelStride == 2) {
            // NV21 format: V and U are interleaved
            uValue = uPlane.bytes[uvIndex];
            vValue = vPlane.bytes[uvIndex];
          } else {
            // Planar YUV420
            uValue = uPlane.bytes[uvY * uPlane.bytesPerRow + uvX];
            vValue = vPlane.bytes[uvY * vPlane.bytesPerRow + uvX];
          }

          // YUV to RGB conversion (BT.601 standard)
          final int yVal = yValue - 16;
          final int uVal = uValue - 128;
          final int vVal = vValue - 128;

          int r = ((298 * yVal + 409 * vVal + 128) >> 8).clamp(0, 255);
          int g = ((298 * yVal - 100 * uVal - 208 * vVal + 128) >> 8).clamp(0, 255);
          int b = ((298 * yVal + 516 * uVal + 128) >> 8).clamp(0, 255);

          image.setPixelRgb(x, y, r, g, b);
        }
      }

      // Apply rotation based on sensor orientation
      if (_sensorOrientation == 90) {
        image = img.copyRotate(image, angle: 90);
      } else if (_sensorOrientation == 180) {
        image = img.copyRotate(image, angle: 180);
      } else if (_sensorOrientation == 270) {
        image = img.copyRotate(image, angle: 270);
      }

      // Mirror for front camera
      if (_currentLensDirection == CameraLensDirection.front) {
        image = img.flipHorizontal(image);
      }

      // Encode to JPEG
      return Uint8List.fromList(img.encodeJpg(image, quality: 85));
    } catch (e) {
      debugPrint('Error converting YUV420: $e');
      return null;
    }
  }

  Uint8List? _convertBGRA8888ToJpeg(CameraImage cameraImage) {
    try {
      final int width = cameraImage.width;
      final int height = cameraImage.height;
      final plane = cameraImage.planes[0];

      final image = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int index = y * plane.bytesPerRow + x * 4;
          final int b = plane.bytes[index];
          final int g = plane.bytes[index + 1];
          final int r = plane.bytes[index + 2];

          image.setPixelRgb(x, y, r, g, b);
        }
      }

      return Uint8List.fromList(img.encodeJpg(image, quality: 85));
    } catch (e) {
      debugPrint('Error converting BGRA8888: $e');
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Live Detection'),
        actions: [
          if (_isStreaming)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Center(
                child: Text(
                  '${_currentFps.toStringAsFixed(1)} FPS',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ),
            ),
          // Camera switch button
          if (cameras.length > 1)
            IconButton(
              onPressed: _cameraController?.value.isInitialized == true
                  ? _switchCamera
                  : null,
              icon: Icon(
                _currentLensDirection == CameraLensDirection.back
                    ? Icons.camera_front
                    : Icons.camera_rear,
              ),
              tooltip: 'Switch camera',
            ),
        ],
      ),
      body: Column(
        children: [
          // Model status indicator
          if (!_isModelLoaded)
            Container(
              padding: const EdgeInsets.all(12),
              margin: const EdgeInsets.all(16),
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
                  Text('Loading YOLO model...'),
                ],
              ),
            ),

          // Camera preview with detections
          Expanded(
            child: _cameraController != null && _cameraController!.value.isInitialized
                ? Stack(
                    fit: StackFit.expand,
                    children: [
                      // Show processed frame with detections if available
                      if (_displayImage != null)
                        CustomPaint(
                          painter: DetectionPainter(
                            image: _displayImage!,
                            results: _detectionResults,
                            selectedIndex: null,
                          ),
                          size: Size.infinite,
                        )
                      else
                        // Show camera preview when not processing
                        CameraPreview(_cameraController!),

                      // Detection count overlay
                      if (_isStreaming && _detectionResults.isNotEmpty)
                        Positioned(
                          top: 16,
                          left: 16,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.black54,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Text(
                              '${_detectionResults.length} object(s)',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                    ],
                  )
                : const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Initializing camera...'),
                      ],
                    ),
                  ),
          ),

          // Controls
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _cameraController?.value.isInitialized == true && _isModelLoaded
                        ? (_isStreaming ? _stopStreaming : _startStreaming)
                        : null,
                    icon: Icon(_isStreaming ? Icons.stop : Icons.play_arrow),
                    label: Text(_isStreaming ? 'Stop' : 'Start Detection'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isStreaming ? Colors.red : null,
                      foregroundColor: _isStreaming ? Colors.white : null,
                    ),
                  ),
                ),
              ],
            ),
          ),

        ],
      ),
    );
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
  double? _lastOcrTimeSeconds;

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
      final stopwatch = Stopwatch()..start();
      final RecognizedText recognizedText =
          await textRecognizer.processImage(inputImage);
      stopwatch.stop();
      final ocrSeconds = stopwatch.elapsedMilliseconds / 1000.0;

      setState(() {
        _recognizedText = recognizedText.text;
        _textBlocks = recognizedText.blocks;
        _lastOcrTimeSeconds = ocrSeconds;
      });

      await textRecognizer.close();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('OCR completed in ${ocrSeconds.toStringAsFixed(1)}s'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
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
