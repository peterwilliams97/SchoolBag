/*
 *  csv.h
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */

class CSV {
    std::vector<std::vector<std::string> >      _rows;
    mutable std::map<const std::string, int>    _index;
    bool                    _has_header_row;
    void   writeRowsToStream(ostream& out) const;
public:
    bool   read(std::string file_path, bool has_header_row);
    bool   write(std::string file_path) const;
    int    getNumRows() const; 
    vector<std::string> getHeader() const;
    const std::string get(int row_num, int col_num) const;
    const std::string get(int row_num, const std::string col_name) const;
};

what !!!