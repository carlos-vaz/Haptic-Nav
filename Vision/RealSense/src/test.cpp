#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

	using namespace cv;
	std::cout << "printing image of cat..." << std::endl;
	Mat img = imread("/Users/Daniel/Desktop/cat.jpg");
	imshow("cat", img);
	std::cout << "press a key to continue to the RS part..." << std::endl;
	waitKey(0);


	// Declare depth colorizer for pretty visualization of depth data
	rs2::colorizer color_map;

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;
	// Start streaming with default recommended configuration
	pipe.start();
	/*rs2::config cfg;
	cfg.enable_device_from_file("/Users/Daniel/Desktop/outdoors.bag");
	pipe.start(cfg); // Load from file
	*/

	const auto window_name = "Display Image";
	namedWindow(window_name, WINDOW_AUTOSIZE);

	while (waitKey(1) < 0) // && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
	{
		rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
		rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
		//rs2::frame color = data.get_color_frame();

		// Query frame size (width and height)
		const int w = depth.as<rs2::video_frame>().get_width();
		const int h = depth.as<rs2::video_frame>().get_height();
	
		// Create OpenCV matrix of size (w,h) from the colorized depth data
		Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
	
		// Update the window with new data
		imshow(window_name, image);
	}


}


