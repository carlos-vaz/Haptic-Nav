#include "tiny_deeplab_api.hpp"
#include <iostream>

int main() {
	using namespace std;

	// Initialize Deeplab object
	Deeplab dl = Deeplab();
	cout << "SUCCESSFULLY CONSTRUCTED DL OBJECT" << endl;

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
	int status = dl.run_segmentation(img_in, seg_out);
	if(status != 0) {
		cout << "Error running segmentation: " << status << endl;
		return 1;
	}
	
	cout << "SUCCESSFULLY RAN SEGMENTATION" << endl;

	// Interpret results
	
	return 0;
}

