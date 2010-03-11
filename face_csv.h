/*
 *  csv.h
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */
#include <string>
#include <vector>
#include <ostream>
#include <map>

const std::string KEY_IMAGE_NAME    = "IMAGE_NAME";
const std::string KEY_FACE_CENTER_X = "FACE_CENTER_X";
const std::string KEY_FACE_CENTER_Y = "FACE_CENTER_Y";
const std::string KEY_FACE_RADIUS   = "FACE_RADIUS";
const std::string KEY_FACE_ANGLE    = "FACE_ANGLE";

class CSV {
    std::vector<std::vector<std::string> >      _rows;
    mutable std::map<const std::string, int>    _index;
    bool                    _has_header_row;
    void   writeRowsToStream(std::ostream& out) const;
public:
    bool   read(std::string file_path, bool has_header_row);
    bool   write(std::string file_path) const;
    int    getNumRows() const; 
    std::vector<std::string> getHeader() const;
    const std::string get(int row_num, int col_num) const;
    const std::string get(int row_num, const std::string col_name) const;
};

