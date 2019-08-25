#ifndef TINY_DEEPLAB_API_HPP_
#define TINY_DEEPLAB_API_HPP_

#include <tensorflow/c/c_api.h>

TF_Buffer* read_file(const char* file);
void free_buffer(void* data, size_t length);
void free_tensor(void* data, size_t length, void* args);

typedef struct segmap {
	const int64_t* dims;
	size_t bytes;
	uint8_t* data_ptr;
} segmap_t;

typedef struct image {
	const int64_t* dims;
	size_t bytes;
	uint8_t* data_ptr;
} image_t;


class Deeplab {
   private:
	TF_Session* session;
	TF_Graph* graph;
	TF_Output output_oper;
	TF_Output input_oper;
	TF_Status* status;

   public:
	Deeplab(); // Constructor 
	~Deeplab();
	int run_segmentation(image_t*, segmap_t*);
};

#endif // TINY_DEEPLAB_API_HPP_

