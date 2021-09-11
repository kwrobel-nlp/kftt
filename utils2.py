import gzip
import jsonlines


def jsonlines_gzip_reader(path, mode='r'):
    gzip_mode = mode + 'b' if 'b' in mode else mode
    fp = gzip.open(path, gzip_mode)
    try:
        fp.read(1)
        fp.seek(0)
        # print('A')
        return jsonlines.Reader(fp)
    except OSError:
        fp.close()
        # print('B')
        return jsonlines.open(path, mode=mode)


def jsonlines_gzip_writer(path, mode='wb'):
    fp = gzip.open(path, mode)
    jsonl_writer = jsonlines.Writer(fp)
    return jsonl_writer
