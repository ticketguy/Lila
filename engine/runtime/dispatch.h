#ifndef LILA_DISPATCH_H
#define LILA_DISPATCH_H

void lila_init_dispatch(void);
void lila_dispatch_matvec(float *out, const float *mat, const float *vec, int rows, int cols);

#endif
