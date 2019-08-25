#include <opencv2/opencv.hpp>
#include "tiny_deeplab_api.hpp"
#include <iostream>
#include <algorithm>

int main() {
	using namespace std;
	using namespace cv;

	// Initialize Deeplab object
	Deeplab dl = Deeplab();
	cout << "Successfully constructed Deeplab object" << endl;

	// Read/resize input image
	Mat image = imread("/Users/Daniel/Desktop/cat.jpg"); 
	int height = image.size().height;
	int width = image.size().width;
	double resize_ratio = (double) 513 / max(height, width);
	Size new_size((int)(resize_ratio*width), (int)(resize_ratio*height));
	Mat resized_image;
	resize(image, resized_image, new_size);
	cout << "Image resized (h, w): (" << height << "," << width << ") --> (" << new_size.height << ", " << new_size.width << ")" << endl;
	imshow("Image", resized_image);
	waitKey(0);

	// Allocate input image object
	const int64_t dims_in[3] = {513, 513, 3};
	image_t* img_in = (image_t*)malloc(sizeof(image_t));
	img_in->dims = &dims_in[0];
	img_in->data_ptr = (uint8_t*)malloc(513*513*3);
	img_in->bytes = 513*513*3*sizeof(uint8_t);

	// Allocate output segmentation map object
	const int64_t dims_out[2] = {513, 513};
	segmap_t* seg_out = (segmap_t*)malloc(sizeof(segmap_t));
	seg_out->dims = &dims_out[0];
	seg_out->data_ptr = (uint8_t*)malloc(513*513);
	seg_out->bytes = 513*513*sizeof(uint8_t);
	
	// Run Deeplab
	cout << "Running segmentation" << endl;
	int status = dl.run_segmentation(img_in, seg_out);
	if(status != 0) {
		cout << "ERROR RUNNING SEGMENTATION: " << status << endl;
		return 1;
	}
	
	cout << "Successfully ran segmentation" << endl;

	// Interpret results

	return 0;
}

