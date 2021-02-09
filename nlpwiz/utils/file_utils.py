import csv
import traceback

def read_csv(file_path, has_header = True):
    data = []
    with open(file_path, 'U') as f:
        if has_header: f.readline( )
        csv_reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
        try:
            for line in csv_reader:
                data.append(line)
        except Exception as e:
            traceback.print_exc()
            print(('file %s, line %d: %s' % (file_path, csv_reader.line_num, e)))
            #sys.exit('file %s, line %d: %s' % (file_path, reader.line_num, e))
    return data
