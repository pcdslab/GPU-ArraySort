this algorithm is able to sort large number of same-sized arrays on a GPU. 
It utilizes CUDA's two layered parallelism in a very efficient manner, by
moving the excessively used data in the on-chip shared memory while keeping
the remaining in global memory. It is effectively an in-place algorithm. 
Compilation information for this algorithm is provided in the source code file.

If you use GPU-ArraySort, please site our work using :

Bibtex:

@inproceedings{AwanSaeed2016Sort,
  title={GPU-ArraySort: A parallel, in-place algorithm for sorting large number of arrays},
  author={Awan, Muaaz and Saeed, Fahad},
  booktitle={Proceedings of Workshop on High Performance Computing for Big Data, International Conference on Parallel Processing (ICPP-2016), Philadelphia PA},
  pages={1--10},
  year={2016}
}


MLA:
Awan, Muaaz, and Fahad Saeed. "GPU-ArraySort: A parallel, in-place algorithm for sorting large number of arrays.",
Proceedings of Workshop on High Performance Computing for Big Data, International Conference on Parallel Processing 
(ICPP-2016), Philadelphia PA, (2016).