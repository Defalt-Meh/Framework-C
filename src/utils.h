#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h> 
#include <errno.h>
#include <stdbool.h> 
#include <ctype.h>  // for isspace

typedef struct{
    float ** in;    /*2D array for inputs*/
    float ** tg;    /*2D array for targets*/
    int nips;       /*Number of Inputs*/
    int nops;       /*Number of Outputs*/
    int rows;       /*Number of rows*/
}Data;

/*Prototype Functions*/
char * readln(FILE * const file);
float ** new2d(const int rows, const int cols);
Data ndata(const int nips, const int nops, const int rows);
void parse(const Data data, char * line, const int rows);
void dfree(Data *d);
void shuffle(const Data d); 
Data build(const char * path, const int nips, const int nops);
int lns(FILE * const file);

#endif  /* UTILS_H */