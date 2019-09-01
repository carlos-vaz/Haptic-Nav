#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "tiny_deeplab_api.hpp"

int main() {
	using namespace cv;
	using namespace std;

	rs2::colorizer color_map;
	rs2::pipeline pipe;
	pipe.start();

	namedWindow("Original", WINDOW_AUTOSIZE);
	namedWindow("Segmented", WINDOW_AUTOSIZE);
	namedWindow("Depth", WINDOW_AUTOSIZE);

	Deeplab dl = Deeplab();

	while(1) 
	{
		// Display depth frame
		rs2::frameset data = pipe.wait_for_frames(); 
		rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
		const int w_d = depth.as<rs2::video_frame>().get_width();
		const int h_d = depth.as<rs2::video_frame>().get_height();
		Mat depth_cv(Size(w_d, h_d), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);
		imshow("Depth", depth_cv);

		// Display RGB frame
		rs2::video_frame original = data.get_color_frame();
		const int w_o = original.get_width();
		const int h_o = original.get_height();
		Mat original_cv(Size(w_o, h_o), CV_8UC3, (void*)original.get_data(), Mat::AUTO_STEP);
		imshow("Original", original_cv);

		// Resize RGB frame
		int orig_height = original_cv.size().height;
		int orig_width = original_cv.size().width;
		double resize_ratio = (double) 513 / max(orig_height, orig_width);
		Size new_size((int)(resize_ratio*orig_width), (int)(resize_ratio*orig_height));
		Mat resized_image;
		resize(original_cv, resized_image, new_size);
		cout << "Image resized (h, w): (" << orig_height << "," << orig_width << ") --> (" << new_size.height << ", " << new_size.width << ")" << endl;

		// Prepare image_t and segmap_t objects before calling Deeplab
		const int64_t dims_in[4] = {1, new_size.height, new_size.width, 3};
		image_t* img_in = (image_t*)malloc(sizeof(image_t));
		img_in->dims = &dims_in[0];
		img_in->data_ptr = resized_image.data;
		img_in->bytes = new_size.width*new_size.height*3*sizeof(uint8_t);

		const int64_t dims_out[3] = {1, new_size.height, new_size.width};
		segmap_t* seg_out = (segmap_t*)malloc(sizeof(segmap_t));
		seg_out->dims = &dims_out[0];
		seg_out->bytes = new_size.width*new_size.height*sizeof(int64_t);

		// Call Deeplab
		int status = dl.run_segmentation(img_in, seg_out);
		if(status != 0) {
			cout << "ERROR RUNNING SEGMENTATION: " << status << endl;
			return 1;
		}

		// Display results
		Mat mySeg(new_size.height, new_size.width, CV_8UC1);
		for(int i=0; i<new_size.width*new_size.height; i++){
			mySeg.data[i] = 5*(uint8_t)seg_out->data_ptr[i];
		}
		imshow("Segmented", mySeg);
		waitKey(0);

	}

}

