#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <tensorflow/c/c_api.h>
#include "tiny_deeplab_api.hpp"

#define SIZE 513

Deeplab::Deeplab() {
	using namespace std;
	cout << "Hello from TensorFlow C library version" << TF_Version() << endl;

	// Import Deeplab graph (as a frozen graph, it has the weights hard-coded in as constants, so no need to restore the checkpoint)
	TF_Buffer* graph_def = read_file("../Models/Deeplab_model_unpacked/deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb");
	graph = TF_NewGraph();
	status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, graph_def, opts, status);
	TF_DeleteImportGraphDefOptions(opts);
	if (TF_GetCode(status) != TF_OK) {
		fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
		return;
	}
	cout << "Successfully loaded Deeplab graph" << endl;
	TF_DeleteBuffer(graph_def);

	// Initialize Session
	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	session = TF_NewSession(graph, sess_opts, status);
}

Deeplab::~Deeplab() {
	using namespace std;
	TF_CloseSession(session, status);
	TF_DeleteSession(session, status);
	TF_DeleteStatus(status);
	TF_DeleteGraph(graph);
	cout << "DESTROYED DEEPLAB OBJECT" << endl;
}

int Deeplab::run_segmentation(image_t* img, segmap_t* seg) {

	// Allocate the input tensor
	TF_Tensor* input = TF_NewTensor(TF_UINT8, img->dims, 3, img->data_ptr, img->bytes, &free_tensor, NULL);
	TF_Operation* oper_in = TF_GraphOperationByName(graph, "ImageTensor");
	TF_Output oper_in_ = {oper_in, 0};

	// Allocate the output tensor
	TF_Tensor* output = TF_NewTensor(TF_UINT8, seg->dims, 2, seg->data_ptr, 513*513, &free_tensor, NULL);
	TF_Operation* oper_out = TF_GraphOperationByName(graph, "SemanticPredictions");
	TF_Output oper_out_ = {oper_out, 0};

	// Run the session on the input tensor
	TF_SessionRun(session, nullptr, &oper_in_, &input, 1, &oper_out_, &output, 1, nullptr, 0, nullptr, status);

	return 0; 
}

TF_Buffer* read_file(const char* file) {                                                  
	FILE *f = fopen(file, "rb");
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);  //same as rewind(f);

	void* data = malloc(fsize);
	fread(data, fsize, 1, f);
	fclose(f);

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = free_buffer; 
	return buf;
}

void free_buffer(void* data, size_t length) { 
        free(data);
}

void free_tensor(void* data, size_t length, void* args) { 
        free(data);
}
