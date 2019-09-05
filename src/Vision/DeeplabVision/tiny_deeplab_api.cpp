#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <tensorflow/c/c_api.h>
#include "tiny_deeplab_api.hpp"

Deeplab::Deeplab() {
	using namespace std;
	cout << "Hello from TensorFlow C library version " << TF_Version() << endl;

	#ifndef PATH_TO_MODELS_DIR
	cout << "ERROR DURING BUILD: PATH TO MODELS DIRECTORY WAS NOT SPECIFIED BY CMAKE" << endl;
	return 1;
	#endif 

	// Import Deeplab graph (as a frozen graph, it has the weights hard-coded in as constants, so no need to restore the checkpoint)
	string path = "/deeplabv3_cityscapes.pb";
	path = PATH_TO_MODELS_DIR + path;
	TF_Buffer* graph_def = read_file(path.data());
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
	cout << "TF_NewSession status: " << TF_GetCode(status) << endl;
	curr_iTensor = NULL;
	curr_oTensor = NULL;
}

Deeplab::~Deeplab() {
	using namespace std;
	TF_CloseSession(session, status);
	TF_DeleteSession(session, status);
	TF_DeleteStatus(status);
	TF_DeleteGraph(graph);
	cout << "Destroyed Deeplab object" << endl;
}

int Deeplab::run_segmentation(image_t* img, segmap_t* seg) {

	// Allocate the input tensor
	TF_Tensor* const input = TF_NewTensor(TF_UINT8, img->dims, 4, img->data_ptr, img->bytes, &free_tensor, NULL);
	TF_Operation* oper_in = TF_GraphOperationByName(graph, "ImageTensor");
	const TF_Output oper_in_ = {oper_in, 0};

	// Allocate the output tensor
	TF_Tensor * output = TF_AllocateTensor(TF_UINT8, seg->dims, 3, seg->bytes);
	TF_Operation* oper_out = TF_GraphOperationByName(graph, "SemanticPredictions");
	const TF_Output oper_out_ = {oper_out, 0};

	// Run the session on the input tensor
	printf("\n########\nsession: %lx\noper_in_: %lx\ninput: %lx\noper_out_: %lx\noutput: %lx\n########\n\n", session, oper_in_.oper, input, oper_out_.oper, output);
	TF_SessionRun(session, nullptr, &oper_in_, &input, 1, &oper_out_, &output, 1, nullptr, 0, nullptr, status);
	seg->data_ptr = static_cast<int64_t*>(TF_TensorData(output));

	curr_iTensor = input;
	curr_oTensor = output;
	return TF_GetCode(status); // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_status.h#L42 
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
	using namespace std;
	cout << "FREEING A TENSOR" << endl;
        free(data);
}
