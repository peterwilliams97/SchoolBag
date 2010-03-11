/*
 *  face_util.cpp
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */

// #include <iostream>
#include <sstream>

#include "face_util.h"

using namespace std;

string intToStr(int n) {
    stringstream s;
    s << n;
    return s.str();
}

string doubleToStr(double n) {
    stringstream s;
    s << n;
    return s.str();
}