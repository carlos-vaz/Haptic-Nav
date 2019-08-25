//TODO: include stuff only once

#ifndef TINY_DEEPLAB_API_HPP_

#include <tensorflow/c/c_api.h>

TF_Buffer* read_file(const char* file);
void free_buffer(void* data, size_t length);
void free_tensor(void* data, size_t length, void* args);

struct segmap {
	const int64_t dims[2];
	
};

struct image {
	const int64_t dims[3];
};

typedef struct segmap segmap_t;
typedef struct image image_t;

class Deeplab {
   private:
	TF_Session* session;
	TF_Graph* graph;
	TF_Output output_oper;
	TF_Output input_oper;
	TF_Status* status;

   public:
	Deeplab(); // Constructor 
	segmap_t* run_segmentation(image_t*);
};

#endif // TINY_DEEPLAB_API_HPP_
