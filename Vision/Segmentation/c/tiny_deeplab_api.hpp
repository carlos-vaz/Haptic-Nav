//TODO: include stuff only once

#ifndef TINY_DEEPLAB_API_HPP_

#include <tensorflow/c/c_api.h>

TF_Buffer* read_file(const char* file);
void free_buffer(void* data, size_t length);
void free_tensor(void* data, size_t length, void* args);

class Deeplab {
   private:
	TF_Session* session;
	TF_Graph* graph;
	TF_Output output_oper;
	TF_Output input_oper;
	TF_Status* status;

   public:
	Deeplab(); // Constructor 
	int run_segmentation();
};

#endif // TINY_DEEPLAB_API_HPP_

