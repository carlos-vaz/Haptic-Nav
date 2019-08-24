#include "tiny_deeplab_api.hpp"
#include <iostream>

int main() {
	using namespace std;
	Deeplab dl = Deeplab();
	cout << "SUCCESSFULLY CONSTRUCTED DL OBJECT" << endl;
	dl.run_segmentation();
	cout << "SUCCESSFULLY RAN SEGMENTATION" << endl;
	return 0;
}

