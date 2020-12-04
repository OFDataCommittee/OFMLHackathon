#include <stdio.h>
typedef enum{

  red = 255,
  blue = 1

} colors;

void print_enum_c( colors thiscolor ) {

  printf("%d\n", thiscolor);

}

colors set_enum_c( ) {
  return blue;
}