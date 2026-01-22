import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:geolocator/geolocator.dart';

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
    InstanceSegmentationPage(),
    OcrPage(),
    MapPage(),
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
            icon: Icon(Icons.auto_awesome_outlined),
            selectedIcon: Icon(Icons.auto_awesome),
            label: 'Segment',
          ),
          NavigationDestination(
            icon: Icon(Icons.document_scanner_outlined),
            selectedIcon: Icon(Icons.document_scanner),
            label: 'OCR',
          ),
          NavigationDestination(
            icon: Icon(Icons.map_outlined),
            selectedIcon: Icon(Icons.map),
            label: 'Map',
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
// Instance Segmentation Page (RF-DETR ONNX)
// ============================================================================

class InstanceSegmentationPage extends StatefulWidget {
  const InstanceSegmentationPage({super.key});

  @override
  State<InstanceSegmentationPage> createState() => _InstanceSegmentationPageState();
}

class _InstanceSegmentationPageState extends State<InstanceSegmentationPage> {
  File? _selectedImage;
  ui.Image? _decodedImage;
  bool _isProcessing = false;
  bool _isModelLoaded = false;
  final ImagePicker _picker = ImagePicker();
  double? _lastInferenceTimeSeconds;

  OrtSession? _session;
  List<Map<String, dynamic>> _segmentationResults = [];
  int? _selectedResultIndex;

  // Convert logit to probability using sigmoid function
  double _sigmoid(double logit) {
    return 1.0 / (1.0 + math.exp(-logit));
  }

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    debugPrint('=== RF-DETR Model Loading Started ===');
    final loadStopwatch = Stopwatch()..start();

    try {
      // Initialize ONNX Runtime environment
      OrtEnv.instance.init();

      // Load model from assets
      final modelBytes = await rootBundle.load('assets/rfdetr_seg_uint8.onnx');
      final bytes = modelBytes.buffer.asUint8List();

      // Create session options
      final sessionOptions = OrtSessionOptions();

      // Create session from model bytes
      _session = OrtSession.fromBuffer(bytes, sessionOptions);

      loadStopwatch.stop();
      final loadTimeSeconds = loadStopwatch.elapsedMilliseconds / 1000.0;

      // Log model input/output info
      debugPrint('=== RF-DETR Model Info ===');
      debugPrint('Input names: ${_session!.inputNames}');
      debugPrint('Output names: ${_session!.outputNames}');
      debugPrint('Model loading time: ${loadTimeSeconds.toStringAsFixed(3)}s');

      setState(() {
        _isModelLoaded = true;
      });
      debugPrint('RF-DETR ONNX model loaded successfully');
    } catch (e) {
      loadStopwatch.stop();
      debugPrint('Error loading RF-DETR model: $e');
    }
  }

  @override
  void dispose() {
    _session?.release();
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
        _segmentationResults = [];
        _selectedResultIndex = null;
      });
    }
  }

  Future<void> _performSegmentation() async {
    if (_selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select an image first')),
      );
      return;
    }

    if (!_isModelLoaded || _session == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model is still loading...')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _segmentationResults = [];
      _selectedResultIndex = null;
    });

    debugPrint('=== RF-DETR Detection Started ===');
    final totalStopwatch = Stopwatch()..start();

    try {
      // Load and preprocess the image
      final preprocessStopwatch = Stopwatch()..start();
      final imageBytes = await _selectedImage!.readAsBytes();
      final decodedImg = img.decodeImage(imageBytes);

      if (decodedImg == null) {
        throw Exception('Failed to decode image');
      }

      // RF-DETR model expects 432x432 input - resize directly (no aspect ratio preservation)
      const int targetSize = 432;
      final resizedImg = img.copyResize(
        decodedImg,
        width: targetSize,
        height: targetSize,
        interpolation: img.Interpolation.linear,
      );

      // Convert to float tensor with normalization (0-255 to 0-1)
      // RF-DETR expects NCHW format: [1, 3, 432, 432]
      final int channels = 3;
      final int height = targetSize;
      final int width = targetSize;

      // Model expects float input - normalize to 0-1 range
      final inputData = Float32List(1 * channels * height * width);

      int idx = 0;
      // NCHW format: iterate channels first, then height, then width
      for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final pixel = resizedImg.getPixel(x, y);
            if (c == 0) {
              inputData[idx] = pixel.r.toInt() / 255.0;
            } else if (c == 1) {
              inputData[idx] = pixel.g.toInt() / 255.0;
            } else {
              inputData[idx] = pixel.b.toInt() / 255.0;
            }
            idx++;
          }
        }
      }

      // Create input tensor
      final inputShape = [1, channels, height, width];
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        inputData,
        inputShape,
      );

      preprocessStopwatch.stop();
      final preprocessMs = preprocessStopwatch.elapsedMilliseconds;
      debugPrint('Preprocessing time: ${preprocessMs}ms');

      // Get input name from model
      final inputName = _session!.inputNames.first;
      final inputs = {inputName: inputTensor};

      // Run inference
      debugPrint('=== RF-DETR Inference Started ===');
      final inferenceStopwatch = Stopwatch()..start();
      final runOptions = OrtRunOptions();
      final outputs = await _session!.runAsync(runOptions, inputs);
      inferenceStopwatch.stop();
      final inferenceMs = inferenceStopwatch.elapsedMilliseconds;

      // Log raw outputs for debugging
      debugPrint('=== RF-DETR Inference Results ===');
      debugPrint('Inference time: ${inferenceMs}ms');
      debugPrint('Number of outputs: ${outputs?.length ?? 0}');

      if (outputs != null) {
        for (int i = 0; i < outputs.length; i++) {
          final output = outputs[i];
          if (output != null) {
            final outputName = _session!.outputNames[i];
            debugPrint('Output $i ($outputName):');

            if (output is OrtValueTensor) {
              final data = output.value;
              debugPrint('  Data type: ${data.runtimeType}');

              // Print first few values for debugging
              if (data is List) {
                final shape = _inferShape(data);
                debugPrint('  Inferred shape: $shape');
                _printNestedList(data, '  ', maxDepth: 2, maxItems: 5);
              }
            }
          }
        }
      }

      // Parse the outputs - RF-DETR segmentation typically outputs:
      // - boxes/scores/labels for detections
      // - masks for segmentation
      // Parse outputs
      final postprocessStopwatch = Stopwatch()..start();
      final parsedResults = _parseOutputs(outputs, decodedImg.width, decodedImg.height);
      postprocessStopwatch.stop();
      final postprocessMs = postprocessStopwatch.elapsedMilliseconds;

      // Total time
      totalStopwatch.stop();
      final totalMs = totalStopwatch.elapsedMilliseconds;
      final totalSeconds = totalMs / 1000.0;

      debugPrint('Postprocessing time: ${postprocessMs}ms');
      debugPrint('=== RF-DETR Detection Complete ===');
      debugPrint('Total time: ${totalMs}ms (preprocess: ${preprocessMs}ms, inference: ${inferenceMs}ms, postprocess: ${postprocessMs}ms)');
      debugPrint('Parsed ${parsedResults.length} segmentation results');
      for (int i = 0; i < parsedResults.length; i++) {
        final result = parsedResults[i];
        debugPrint('  Result $i: class=${result['className']}, confidence=${result['confidence']}, bbox=${result['boundingBox']}');
      }

      // Clean up
      inputTensor.release();
      runOptions.release();
      outputs?.forEach((output) => output?.release());

      setState(() {
        _segmentationResults = parsedResults;
        _lastInferenceTimeSeconds = totalSeconds;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Segmentation completed in ${totalSeconds.toStringAsFixed(2)}s - ${parsedResults.length} objects found'),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e, stackTrace) {
      totalStopwatch.stop();
      debugPrint('Error during segmentation: $e');
      debugPrint('Stack trace: $stackTrace');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error during segmentation: $e')),
        );
      }
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  List<int> _inferShape(List data) {
    final List<int> shape = [];
    dynamic current = data;
    while (current is List) {
      shape.add(current.length);
      if (current.isNotEmpty) {
        current = current[0];
      } else {
        break;
      }
    }
    return shape;
  }

  void _printNestedList(List data, String indent, {int maxDepth = 2, int maxItems = 5, int currentDepth = 0}) {
    if (currentDepth >= maxDepth) {
      debugPrint('$indent[... ${data.length} items]');
      return;
    }

    final itemsToShow = data.length > maxItems ? maxItems : data.length;
    for (int i = 0; i < itemsToShow; i++) {
      final item = data[i];
      if (item is List) {
        debugPrint('$indent[$i]: List of ${item.length} items');
        _printNestedList(item, '$indent  ', maxDepth: maxDepth, maxItems: maxItems, currentDepth: currentDepth + 1);
      } else {
        debugPrint('$indent[$i]: $item');
      }
    }
    if (data.length > maxItems) {
      debugPrint('$indent... and ${data.length - maxItems} more items');
    }
  }

  List<Map<String, dynamic>> _parseOutputs(List<OrtValue?>? outputs, int originalWidth, int originalHeight) {
    if (outputs == null || outputs.isEmpty) {
      return [];
    }

    final List<Map<String, dynamic>> results = [];

    // RF-DETR output format based on debug logs:
    // - dets: [1, 200, 4] - bounding boxes
    // - labels: [1, 200, 6] - class probabilities (6 classes)
    // - masks: [1, 200, 108, 108] - segmentation masks

    try {
      List<List<List<double>>>? detsData;
      List<List<List<double>>>? labelsData;
      List<List<List<List<double>>>>? masksData;

      for (int i = 0; i < outputs.length; i++) {
        final output = outputs[i];
        if (output == null) continue;

        final outputName = _session!.outputNames[i];
        if (output is OrtValueTensor) {
          final data = output.value;
          final shape = _inferShape(data as List);

          debugPrint('Parsing output $outputName with shape $shape');

          if (outputName == 'dets' || (shape.length == 3 && shape[2] == 4)) {
            // [1, 200, 4] - bounding boxes
            detsData = _convertToDouble3D(data);
          } else if (outputName == 'labels' || (shape.length == 3 && shape[2] == 6)) {
            // [1, 200, 6] - class probabilities
            labelsData = _convertToDouble3D(data);
          } else if (shape.length == 4 && shape[2] == 108 && shape[3] == 108) {
            // [1, 200, 108, 108] - masks
            masksData = _convertToDouble4D(data);
          }
        }
      }

      if (detsData == null || labelsData == null) {
        debugPrint('Could not find dets or labels in outputs');
        return results;
      }

      // Process each of the 200 detections
      final boxes = detsData[0]; // [200, 4]
      final labelProbs = labelsData[0]; // [200, 6]
      final masks = masksData?[0]; // [200, 108, 108] if available

      // Debug: print first few raw boxes and labels to understand the format
      debugPrint('=== Raw detection data (first 5 with highest confidence) ===');
      debugPrint('Masks available: ${masks != null}');

      // First, find detections with confidence and sort
      final List<Map<String, dynamic>> debugDetections = [];
      for (int i = 0; i < boxes.length; i++) {
        final box = boxes[i];
        final probs = labelProbs[i];

        // Find best class
        int bestClass = 0;
        double bestConf = probs[0];
        for (int c = 1; c < probs.length; c++) {
          if (probs[c] > bestConf) {
            bestConf = probs[c];
            bestClass = c;
          }
        }

        debugDetections.add({
          'index': i,
          'box': box,
          'probs': probs,
          'bestClass': bestClass,
          'bestConf': bestConf,
        });
      }

      // Sort by confidence and print top 5
      debugDetections.sort((a, b) => (b['bestConf'] as double).compareTo(a['bestConf'] as double));
      for (int i = 0; i < 5 && i < debugDetections.length; i++) {
        final d = debugDetections[i];
        debugPrint('Detection ${d['index']}: box=${d['box']}, probs=${d['probs']}, bestClass=${d['bestClass']}, bestConf=${d['bestConf']}');
      }
      debugPrint('Original image size: ${originalWidth}x${originalHeight}');

      // Now process detections - skip low confidence
      for (final d in debugDetections) {
        final double bestConf = d['bestConf'] as double;
        final int bestClass = d['bestClass'] as int;
        final List<double> box = d['box'] as List<double>;
        final int detIndex = d['index'] as int;

        // Skip low confidence detections (raw confidence, not percentage)
        if (bestConf < 0.3) continue;

        // Box format needs to be determined - try different interpretations
        // The raw values are very small (0.6, 1.3, 1.2, 2.6) so they might be:
        // - Normalized 0-1 (multiply by image size)
        // - Center x, center y, width, height format
        // - Already in pixel coords for 432x432 input

        // Try: assume box is [cx, cy, w, h] normalized to 0-1
        // Convert to x1, y1, x2, y2 in original image coords
        final double cx = box[0];
        final double cy = box[1];
        final double w = box[2];
        final double h = box[3];

        // If normalized, scale to original image
        final double x1 = (cx - w / 2) * originalWidth;
        final double y1 = (cy - h / 2) * originalHeight;
        final double x2 = (cx + w / 2) * originalWidth;
        final double y2 = (cy + h / 2) * originalHeight;

        debugPrint('Converted box: cx=$cx, cy=$cy, w=$w, h=$h -> x1=$x1, y1=$y1, x2=$x2, y2=$y2');

        // Skip invalid boxes
        if (x2 <= x1 || y2 <= y1) continue;

        // Get the mask for this detection (keep at original 108x108 size for efficiency)
        List<List<double>>? mask;
        if (masks != null && detIndex < masks.length) {
          mask = masks[detIndex]; // 108x108 mask
        }

        results.add({
          'className': _getClassName(bestClass),
          'classIndex': bestClass,
          'confidence': _sigmoid(bestConf),
          'boundingBox': Rect.fromLTRB(x1, y1, x2, y2),
          'mask': mask, // Keep original 108x108 mask, painter will scale it
        });
      }
    } catch (e, stackTrace) {
      debugPrint('Error parsing outputs: $e');
      debugPrint('Stack trace: $stackTrace');
    }

    // Sort by confidence
    results.sort((a, b) => (b['confidence'] as double).compareTo(a['confidence'] as double));

    return results;
  }

  List<List<List<double>>> _convertToDouble3D(List data) {
    return (data).map((batch) {
      return (batch as List).map((item) {
        return (item as List).map((v) => (v as double)).toList();
      }).toList();
    }).toList();
  }

  List<List<List<List<double>>>> _convertToDouble4D(List data) {
    return (data).map((batch) {
      return (batch as List).map((item) {
        return (item as List).map((row) {
          return (row as List).map((v) => (v as double)).toList();
        }).toList();
      }).toList();
    }).toList();
  }

  String _getClassName(int classIndex) {
    // Model class labels are unknown - just show the class index
    // TODO: Add actual class labels if known for the rfdetr_seg_uint8.onnx model
    return 'Class $classIndex';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Instance Segmentation'),
      ),
      body: Column(
        children: [
          // Fixed top section with image and buttons
          Padding(
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
                        Text('Loading RF-DETR model...'),
                      ],
                    ),
                  ),

                // Image display area with segmentation overlay
                Container(
                  height: 280,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: _selectedImage != null && _decodedImage != null
                      ? ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: CustomPaint(
                            painter: SegmentationPainter(
                              image: _decodedImage!,
                              results: _segmentationResults,
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
                const SizedBox(height: 12),

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
                            _isProcessing || !_isModelLoaded ? null : _performSegmentation,
                        icon: _isProcessing
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Icon(Icons.auto_awesome),
                        label: Text(_isProcessing ? 'Processing...' : 'Segment'),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Scrollable results section
          Expanded(
            child: _segmentationResults.isNotEmpty
                ? Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Row(
                          children: [
                            Text(
                              '${_segmentationResults.length} object(s) detected:',
                              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
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
                            if (_lastInferenceTimeSeconds != null)
                              Text(
                                '${_lastInferenceTimeSeconds!.toStringAsFixed(2)}s',
                                style: TextStyle(color: Colors.grey[600]),
                              ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        // Scrollable list of results
                        Expanded(
                          child: ListView(
                            children: _buildResultsSections(),
                          ),
                        ),
                      ],
                    ),
                  )
                : _selectedImage != null && !_isProcessing
                    ? const Center(
                        child: Text(
                          'Press Segment to find and segment objects in the image',
                          style: TextStyle(color: Colors.grey),
                        ),
                      )
                    : const SizedBox.shrink(),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildResultsSections() {
    final List<Widget> widgets = [];

    // Separate Class 1 from other classes, keeping track of original indices
    final List<MapEntry<int, Map<String, dynamic>>> class1Items = [];
    final List<MapEntry<int, Map<String, dynamic>>> otherItems = [];

    for (int i = 0; i < _segmentationResults.length; i++) {
      final result = _segmentationResults[i];
      final classIndex = result['classIndex'] as int;
      if (classIndex == 1) {
        class1Items.add(MapEntry(i, result));
      } else {
        otherItems.add(MapEntry(i, result));
      }
    }

    // Sort Class 1 items by x coordinate (left to right)
    class1Items.sort((a, b) {
      final rectA = a.value['boundingBox'] as Rect;
      final rectB = b.value['boundingBox'] as Rect;
      return rectA.left.compareTo(rectB.left);
    });

    // Sort other items by y coordinate (top to bottom)
    otherItems.sort((a, b) {
      final rectA = a.value['boundingBox'] as Rect;
      final rectB = b.value['boundingBox'] as Rect;
      return rectA.top.compareTo(rectB.top);
    });

    // Build Class 1 section
    if (class1Items.isNotEmpty) {
      widgets.add(
        Padding(
          padding: const EdgeInsets.only(top: 8, bottom: 4),
          child: Text(
            'Class 1 (${class1Items.length}):',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey[700],
            ),
          ),
        ),
      );
      widgets.addAll(class1Items.map((entry) => _buildResultCard(entry.key, entry.value)));
    }

    // Build Other classes section
    if (otherItems.isNotEmpty) {
      widgets.add(
        Padding(
          padding: const EdgeInsets.only(top: 16, bottom: 4),
          child: Text(
            'Other Classes (${otherItems.length}):',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey[700],
            ),
          ),
        ),
      );
      widgets.addAll(otherItems.map((entry) => _buildResultCard(entry.key, entry.value)));
    }

    return widgets;
  }

  Widget _buildResultCard(int index, Map<String, dynamic> result) {
    final confidence = (result['confidence'] as double).clamp(0.0, 1.0);
    final isSelected = _selectedResultIndex == index;
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
        title: Text(result['className'] as String),
        subtitle: Text(
          'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
        ),
        trailing: Icon(
          isSelected ? Icons.visibility : Icons.visibility_outlined,
          color: isSelected ? _getColorForIndex(index) : Colors.grey,
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

// Custom painter to draw segmentation results on the image
class SegmentationPainter extends CustomPainter {
  final ui.Image image;
  final List<Map<String, dynamic>> results;
  final int? selectedIndex;

  SegmentationPainter({
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

    // Draw segmentation masks (only selected one if there's a selection)
    for (int i = 0; i < results.length; i++) {
      // Skip non-selected items if one is selected
      if (selectedIndex != null && selectedIndex != i) {
        continue;
      }

      final result = results[i];
      final color = _getColorForIndex(i);
      final rect = result['boundingBox'] as Rect;
      final mask = result['mask'] as List<List<double>>?;

      // Draw mask if available
      if (mask != null && mask.isNotEmpty) {
        _drawMask(canvas, mask, color, image.width, image.height);
      } else {
        // Fallback to bounding box outline if no mask
        final boxPaint = Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0 / scale;
        canvas.drawRect(rect, boxPaint);
      }

      // Draw label background at top of bounding box
      final labelBgPaint = Paint()
        ..color = color
        ..style = PaintingStyle.fill;

      final confidence = (result['confidence'] as double).clamp(0.0, 1.0);
      final labelText = '${result['className']} ${(confidence * 100).toStringAsFixed(0)}%';
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

  /// Draw a segmentation mask as a semi-transparent overlay
  void _drawMask(Canvas canvas, List<List<double>> mask, Color color, int imgWidth, int imgHeight) {
    final int maskHeight = mask.length;
    final int maskWidth = mask[0].length;

    // Threshold for considering a pixel as part of the mask
    const double threshold = 0.5;

    // Scale factors from mask (108x108) to image coordinates
    final double scaleX = imgWidth / maskWidth;
    final double scaleY = imgHeight / maskHeight;

    // Build a single path for the entire mask (much more efficient than many rectangles)
    final ui.Path fillPath = ui.Path();

    // Process row by row, creating horizontal spans for efficiency
    for (int my = 0; my < maskHeight; my++) {
      int? spanStart;

      for (int mx = 0; mx <= maskWidth; mx++) {
        bool isActive = false;
        if (mx < maskWidth) {
          // Apply sigmoid to convert logits to probabilities
          final double logit = mask[my][mx];
          final double prob = 1.0 / (1.0 + math.exp(-logit));
          isActive = prob > threshold;
        }

        if (isActive && spanStart == null) {
          // Start a new span
          spanStart = mx;
        } else if (!isActive && spanStart != null) {
          // End the span, add rectangle to path
          final double x = spanStart * scaleX;
          final double y = my * scaleY;
          final double w = (mx - spanStart) * scaleX;
          final double h = scaleY;
          fillPath.addRect(Rect.fromLTWH(x, y, w, h));
          spanStart = null;
        }
      }
    }

    // Draw the filled mask with transparency
    final fillPaint = Paint()
      ..color = color.withOpacity(0.4)
      ..style = PaintingStyle.fill;
    canvas.drawPath(fillPath, fillPaint);

    // Draw outline around the mask
    final outlinePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawPath(fillPath, outlinePaint);
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
  bool shouldRepaint(covariant SegmentationPainter oldDelegate) {
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

// ============================================================================
// Map Page
// ============================================================================

class MapPage extends StatefulWidget {
  const MapPage({super.key});

  @override
  State<MapPage> createState() => _MapPageState();
}

// Data class for Munich points
class MunichPoint {
  final String id;
  final LatLng position;

  MunichPoint({
    required this.id,
    required this.position,
  });
}

class _MapPageState extends State<MapPage> {
  final MapController _mapController = MapController();
  bool _isLoading = false;
  bool _dataLoaded = false;
  Position? _userPosition;
  final List<MunichPoint> _munichPoints = [];
  final List<Marker> _markers = [];
  final math.Random _random = math.Random();
  MunichPoint? _selectedPoint;
  OverlayEntry? _tooltipOverlay;

  // Munich center coordinates
  static const double _munichCenterLat = 48.1351;
  static const double _munichCenterLng = 11.5820;
  static const double _munichRadius = 0.15; // Approximate radius in degrees

  @override
  void initState() {
    super.initState();
    // Load data when the page is first created
    _loadData();
  }

  Future<void> _loadData() async {
    if (_dataLoaded) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // Generate 50 random points in Munich
      _generateMunichPoints();

      // Get user location
      await _getUserLocation();

      // Create markers
      _createMarkers();

      setState(() {
        _dataLoaded = true;
        _isLoading = false;
      });
    } catch (e) {
      debugPrint('Error loading map data: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error loading map data: $e')),
        );
      }
      setState(() {
        _isLoading = false;
      });
    }
  }

  String _generateRandomId() {
    // Generate a random ID (e.g., "PT-12345")
    final int randomNum = _random.nextInt(99999);
    return 'PT-${randomNum.toString().padLeft(5, '0')}';
  }

  void _generateMunichPoints() {
    _munichPoints.clear();
    for (int i = 0; i < 50; i++) {
      // Generate random point within Munich area
      // Using a simple approach: random offset from center
      final double latOffset = (_random.nextDouble() - 0.5) * 2 * _munichRadius;
      final double lngOffset = (_random.nextDouble() - 0.5) * 2 * _munichRadius;
      
      final double lat = _munichCenterLat + latOffset;
      final double lng = _munichCenterLng + lngOffset;
      
      final String id = _generateRandomId();
      _munichPoints.add(MunichPoint(
        id: id,
        position: LatLng(lat, lng),
      ));
    }
    debugPrint('Generated ${_munichPoints.length} random points in Munich');
  }

  Future<void> _getUserLocation() async {
    try {
      // Check if location services are enabled
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        debugPrint('Location services are disabled');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Location services are disabled. Please enable them in settings.'),
            ),
          );
        }
        return;
      }

      // Check location permissions
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          debugPrint('Location permissions are denied');
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Location permissions are denied. Please enable them in settings.'),
              ),
            );
          }
          return;
        }
      }

      if (permission == LocationPermission.deniedForever) {
        debugPrint('Location permissions are permanently denied');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Location permissions are permanently denied. Please enable them in app settings.'),
            ),
          );
        }
        return;
      }

      // Get current position
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      setState(() {
        _userPosition = position;
      });
      debugPrint('User location: ${position.latitude}, ${position.longitude}');
    } catch (e) {
      debugPrint('Error getting user location: $e');
      // Don't show error to user if location fails - just continue without it
    }
  }

  void _createMarkers() {
    _markers.clear();

    // Add markers for Munich points
    for (int i = 0; i < _munichPoints.length; i++) {
      final munichPoint = _munichPoints[i];
      _markers.add(
        Marker(
          point: munichPoint.position,
          width: 60,
          height: 60,
          child: Icon(
            Icons.location_on,
            color: Colors.blue,
            size: 60,
          ),
        ),
      );
    }

    // Add marker for user location if available
    if (_userPosition != null) {
      final userLatLng = LatLng(_userPosition!.latitude, _userPosition!.longitude);
      _markers.add(
        Marker(
          point: userLatLng,
          width: 70,
          height: 70,
          child: Icon(
            Icons.my_location,
            color: Colors.red,
            size: 70,
          ),
        ),
      );
    }
  }

  void _showTooltip(MunichPoint point) {
    setState(() {
      _selectedPoint = point;
    });

    // Remove existing tooltip if any
    _hideTooltip();

    // Show tooltip using a dialog or overlay
    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (BuildContext context) {
        return Dialog(
          backgroundColor: Colors.transparent,
          insetPadding: const EdgeInsets.all(20),
          child: Stack(
            children: [
              Positioned(
                child: Material(
                  color: Colors.transparent,
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.2),
                          blurRadius: 10,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'Point Information',
                              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                            ),
                            IconButton(
                              icon: const Icon(Icons.close, size: 20),
                              onPressed: () {
                                Navigator.of(context).pop();
                                setState(() {
                                  _selectedPoint = null;
                                });
                              },
                              padding: EdgeInsets.zero,
                              constraints: const BoxConstraints(),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        _buildInfoRow('ID', point.id),
                        _buildInfoRow('Latitude', point.position.latitude.toStringAsFixed(6)),
                        _buildInfoRow('Longitude', point.position.longitude.toStringAsFixed(6)),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    ).then((_) {
      setState(() {
        _selectedPoint = null;
      });
    });
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              '$label:',
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: Colors.grey[700],
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _hideTooltip() {
    if (_tooltipOverlay != null) {
      _tooltipOverlay!.remove();
      _tooltipOverlay = null;
    }
  }

  void _fitBounds() {
    if (_markers.isEmpty && _munichPoints.isEmpty) return;

    // Collect all points
    final List<LatLng> allPoints = _munichPoints.map((p) => p.position).toList();
    if (_userPosition != null) {
      allPoints.add(LatLng(_userPosition!.latitude, _userPosition!.longitude));
    }

    if (allPoints.isEmpty) return;

    // Calculate bounds to include all points
    double minLat = double.infinity;
    double maxLat = -double.infinity;
    double minLng = double.infinity;
    double maxLng = -double.infinity;

    for (final point in allPoints) {
      minLat = math.min(minLat, point.latitude);
      maxLat = math.max(maxLat, point.latitude);
      minLng = math.min(minLng, point.longitude);
      maxLng = math.max(maxLng, point.longitude);
    }

    // Calculate center and zoom level
    final center = LatLng(
      (minLat + maxLat) / 2,
      (minLng + maxLng) / 2,
    );

    // Calculate zoom level based on bounds
    final latDiff = maxLat - minLat;
    final lngDiff = maxLng - minLng;
    final maxDiff = math.max(latDiff, lngDiff);
    
    // Approximate zoom calculation
    double zoom = 12.0;
    if (maxDiff > 0.1) {
      zoom = 10.0;
    } else if (maxDiff > 0.05) {
      zoom = 11.0;
    } else if (maxDiff > 0.02) {
      zoom = 12.0;
    } else if (maxDiff > 0.01) {
      zoom = 13.0;
    } else {
      zoom = 14.0;
    }

    _mapController.move(center, zoom);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Map'),
        actions: [
          if (_dataLoaded)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: () {
                setState(() {
                  _dataLoaded = false;
                });
                _loadData();
              },
              tooltip: 'Reload data',
            ),
        ],
      ),
      body: Stack(
        children: [
          // Map
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: LatLng(_munichCenterLat, _munichCenterLng),
              initialZoom: 12.0,
              onMapReady: () {
                // If data is loaded, fit bounds to show all markers
                if (_dataLoaded && _markers.isNotEmpty) {
                  WidgetsBinding.instance.addPostFrameCallback((_) {
                    _fitBounds();
                  });
                }
              },
              onTap: (tapPosition, point) {
                // Check if tap is on a Munich point marker
                // Use a threshold in degrees (approximately 150 meters for larger markers)
                const double threshold = 0.0015;
                MunichPoint? closestPoint;
                double closestDistance = double.infinity;

                for (final munichPoint in _munichPoints) {
                  // Calculate simple distance in degrees
                  final double latDiff = (point.latitude - munichPoint.position.latitude).abs();
                  final double lngDiff = (point.longitude - munichPoint.position.longitude).abs();
                  final double distance = math.sqrt(latDiff * latDiff + lngDiff * lngDiff);
                  
                  if (distance < threshold && distance < closestDistance) {
                    closestPoint = munichPoint;
                    closestDistance = distance;
                  }
                }

                if (closestPoint != null) {
                  _showTooltip(closestPoint);
                }
              },
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'com.example.flutter_cv_edge_app',
              ),
              MarkerLayer(
                markers: _markers,
              ),
            ],
          ),
          // Loading indicator
          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.3),
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text(
                      'Loading map data...',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          // Info overlay
          if (_dataLoaded && !_isLoading)
            Positioned(
              top: 16,
              left: 16,
              right: 16,
              child: Card(
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'Map Data Loaded',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 4),
                      Text('Munich points: ${_munichPoints.length}'),
                      Text(
                        _userPosition != null
                            ? 'Your location: ${_userPosition!.latitude.toStringAsFixed(4)}, ${_userPosition!.longitude.toStringAsFixed(4)}'
                            : 'Your location: Not available',
                      ),
                      if (_selectedPoint != null) ...[
                        const SizedBox(height: 4),
                        Text(
                          'Selected: ${_selectedPoint!.id}',
                          style: TextStyle(
                            color: Colors.blue[700],
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
      floatingActionButton: _dataLoaded && !_isLoading
          ? FloatingActionButton(
              onPressed: _fitBounds,
              tooltip: 'Fit all markers',
              child: const Icon(Icons.fit_screen),
            )
          : null,
    );
  }

  @override
  void dispose() {
    _hideTooltip();
    _mapController.dispose();
    super.dispose();
  }
}
