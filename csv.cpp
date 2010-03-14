/*
 *  csv.cpp
 *  FaceTracker
 *
 *  Created by peter on 11/03/10.
 */
 
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include "face_csv.h"

using namespace std;

static string trim(const string& in) {
    string out = "";
    string whitespace = " \t\r\n";
    string::size_type p2 = in.find_last_not_of(whitespace);
    if (p2 != string::npos) {
         string::size_type p1 = in.find_first_not_of(whitespace);
        if (p1 == string::npos) p1 = 0;
        out = in.substr(p1, (p2-p1)+1);
    }
    return out;
}

static vector<string> readCsvRow(string line) {
    const string delimiters = ",";
    vector<string> row;
    line = trim(line);
    if (line.size() > 0) {
       string::size_type last_pos = 0;
        while (last_pos != string::npos) {
            string val;
            string::size_type pos = line.find_first_of(delimiters, last_pos);
            val = trim(line.substr(last_pos, pos - last_pos));
            if (pos != string::npos || val.size() > 0) {  // Not last in line or not empty
                const string cval(val);
                row.push_back(cval);
            }
            if (pos == string::npos)
                break;
            last_pos = line.find_first_not_of(delimiters, pos);
        }
    }
    return row;
}


/*
 *  Read a CSV file. Comma separated. One file per line
 */
bool readCsvFile(vector<vector<string> >& rows, const string files_path, bool has_header_row) {
    rows.resize(0);
    bool ok = true;
    ifstream input_file;
    input_file.open(files_path.c_str(), fstream::in);
    if (!input_file.is_open()) {
        cerr << "Could not open " << files_path << endl;
        ok = false;
    }
    else {
        string line;
        while (getline(input_file, line)) {
            vector<string> row = readCsvRow(line);
            if (row.size() > 0)
                rows.push_back(row);
        }
        input_file.close();
        if (has_header_row && rows.size() >= 1) {
            int num_cols = rows[0].size();
            for (int i = 1; i < (int)rows.size(); i++)
                rows[i].resize(num_cols);
        }       
    }
    return ok;
}

/*
 *  Write CSV file contents to output
 */
#if 0
static void writeCsvFileContents(const vector<vector<string> >& rows, ostream& output) {
    for (vector<vector<string> >::const_iterator ir = rows.begin(); ir != rows.end(); ir++) {
        const vector<string>& row = *ir;
        for (vector<string>::const_iterator iv = row.begin(); iv != row.end(); iv++) {
            output << *iv << ", ";
        }
        output << endl;
    }
}
#endif

/*
 *  Write a list of files and settings. Comma separated. One file per line
 */
 #if 0
static bool writeCsvFile(const vector<vector<string> >& rows, const string file_path) {
    bool ok = true;
    if (file_path.size() == 0) {
        writeCsvFileContents(rows, cout);
    }
    else {
        ofstream output_file;
        output_file.open(file_path.c_str(), fstream::in);
        if (!output_file.is_open()) {
            cerr << "Could not open " << file_path << endl;
            ok = false;
        }
        else {
            writeCsvFileContents(rows, cout);
            output_file.close();
        }
    }
    return ok;
}
#endif

bool CSV::read(string file_path, bool has_header_row) {
    _has_header_row = has_header_row;
    _rows.resize(0);
  
    bool ok = readCsvFile(_rows,  file_path, _has_header_row);
    if (ok && _has_header_row && _rows.size() >= 1) {
        vector<string> header_row = _rows[0];
        for (int i = 0; i < (int)header_row.size(); i++) {
            _index.insert(make_pair<const string, int>(header_row[i], i));
        }
    }
    return ok;
}

void CSV::writeRowsToStream(ostream& out) const  {
    if (_has_header_row && _rows.size() >= 1) {
        vector<string> header_row = _rows[0];
        vector<string>::const_iterator it;
        for (it = header_row.begin(); it != header_row.end(); it++)
            out << *it << ", ";
        out << endl;
        cerr << "getNumRows() = " << getNumRows() << endl;
        for (int row_num = 0; row_num < getNumRows(); row_num++) {
            for (it = header_row.begin(); it != header_row.end(); it++)
                out << get(row_num, *it) << ", ";
            out << endl;
        }
    }
}


bool CSV::write(string file_path) const {
    bool ok = true;
    ofstream output_file;
       
    if (file_path.size() > 0) {
        output_file.open(file_path.c_str(), fstream::in);
        if (!output_file.is_open()) {
            cerr << "Could not open " << file_path << endl;
            ok = false;
        }
    }
    writeRowsToStream(output_file.is_open() ? output_file : cout);
    if (output_file.is_open())
        output_file.close();
    return ok;
}

int  CSV::getNumRows() const { 
    return (_has_header_row && _rows.size() > 0) ? _rows.size() -1 : _rows.size();
}

vector<string> CSV::getHeader() const {
    vector<string> header_row(0);
    if (_has_header_row && _rows.size() >= 1)
        header_row = _rows[0];
    return header_row;
}

const string CSV::get(int row_num, int col_num) const  {
    int n = _has_header_row ? row_num + 1 : row_num;
    return _rows[n][col_num];
}

const string CSV::get(int row_num,  const string col_name) const  {
    string val = "";
    int col_num;
    if (_has_header_row) {
        col_num = _index[col_name];
        val = get(row_num, col_num);
    } 
    else {
        cerr << "Cannot be here !" << endl;
        exit(55);
    }
    if (col_name.compare(KEY_IMAGE_NAME) == 0 && val.compare("") == 0)
        cerr << "Cannot be here ! row_num = " << row_num << ", col_name = " << col_name << endl;

//    cerr << "get(" << row_num << ", " << col_name << ") = " << val << endl;
    return val;
}

