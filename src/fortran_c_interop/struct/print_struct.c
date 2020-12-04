#include <stdio.h>
#include <string.h>

struct foo {
   int    id;
   char   key[10];
   float  pi;
   double one;
};

void print_foo_value( struct foo in ){

  printf("%d\n", in.id);
  printf("%s\n", in.key);
  printf("%f\n", in.pi);
  printf("%g\n", in.one);

}

void print_foo_ptr( struct foo *in ){

  printf("%d\n", in->id);
  printf("%s\n", in->key);
  printf("%f\n", in->pi);
  printf("%g\n", in->one);

}