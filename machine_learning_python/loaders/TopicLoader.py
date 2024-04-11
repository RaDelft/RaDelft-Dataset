import os
import yaml
import json
import bisect
import glob
import pickle

from loaders.messages import get_msg_class

TOPIC_FILE_EXTENSION = '.topic.json'

class TopicIndex(object):
    def __init__(self, name, ds_dir, msg_type=None):
        self.name = name
        self.ds_dir = ds_dir
        self.msg_type = msg_type

        out_txt_filename = self.name + TOPIC_FILE_EXTENSION
        self.topic_filepath = os.path.join(self.ds_dir, out_txt_filename)

        self.MsgClass = None if (self.msg_type) else get_msg_class(self.msg_type)

        self.all_messages = []
        self.all_timestamps_ns = []

    def __load_index(self):

        cache_dir = os.path.join(self.ds_dir, '.cache')
        cache_filename = self.name + '.topic.pkl'
        cache_filepath = os.path.join(cache_dir, cache_filename)

        index = None
        try:
            cache_mtime = os.path.getmtime(cache_filepath)
            index_mtime = os.path.getmtime(self.topic_filepath)
            if (cache_mtime > index_mtime):
                # cache is up to date, use fast cache load
                index = pickle.load(open(cache_filepath, 'rb'))
        except (OSError, ValueError):
            # could not find cache; no problem
            pass

        if index is None:
            print('Update cache file %s ...' % cache_filepath)

            # NOTE: we use the yaml library here instead of json
            # since json is a subset of yaml, and yaml.safe_load
            # returns str instead of unicode
            # See https://stackoverflow.com/a/16373377/218682
            # index = yaml.safe_load(open(self.topic_filepath))
            index = json.load(open(self.topic_filepath))

            # update cache
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            pickle.dump(index, open(cache_filepath, 'wb'))

        return index

    def get_all_timestamp_ns(self):
        return self.all_timestamps_ns

    def get_time_range(self):
        # ASSERT: assumes time stamps are ordered
        return (self.all_timestamps_ns[0], self.all_timestamps_ns[-1])

    def get_index_from_timestamp_ns(self, timestamp_ns):
        pos = bisect.bisect_left(self.all_timestamps_ns, timestamp_ns)
        if pos == len(self.all_timestamps_ns):
            return pos - 1
        return pos

    def get_message(self, timestamp_ns):
        idx = self.get_index_from_timestamp_ns(timestamp_ns)
        return self.all_messages[idx]

    def get_messages_in_range(self, timestamp_ns_range):
        first_timestamp_ns, last_timestamp_ns = timestamp_ns_range

        # find first message at the start of the given range
        first_idx = self.get_index_from_timestamp_ns(first_timestamp_ns)
        first_ts = self.all_timestamps_ns[first_idx]
        if first_ts < first_timestamp_ns:
            first_idx += 1

        # find first message after the range (which should not be included)
        last_idx = self.get_index_from_timestamp_ns(last_timestamp_ns)
        last_idx += 1

        # return messages in the included range
        return self.all_messages[first_idx:last_idx]

    def get_temporally_closest_index(self, timestamp_ns):
        pos = bisect.bisect_left(self.all_timestamps_ns, timestamp_ns)
        if pos == len(self.all_timestamps_ns):
            return pos - 1
        if abs(self.all_timestamps_ns[pos - 1] - timestamp_ns) < abs(self.all_timestamps_ns[pos] - timestamp_ns):
            return pos - 1
        else:
            return pos

    def get_temporally_closest_message(self, timestamp_ns):
        idx = self.get_temporally_closest_index(timestamp_ns)
        return self.all_messages[idx]

    def add_message(self, msg):
        ts = msg.get_timestamp_ns()

        if ((len(self.all_messages) == 0) or (self.all_timestamps_ns[-1] <= ts)):
            # common case, append to end
            self.all_messages.append(msg)
            self.all_timestamps_ns.append(ts)
        else:
            # find index such that all
            #   all_messages[:pos] < ts, and all_messages[pos:] >= ts
            pos = bisect.bisect_right(self.all_timestamps_ns, ts)
            self.all_messages = self.all_messages[:pos] + [msg] + self.all_messages[pos:]
            self.all_timestamps_ns = self.all_timestamps_ns[:pos] + [ts] + self.all_timestamps_ns[pos:]

    def __len__(self):
        return len(self.all_messages)

    def clear_messages(self):
        self.all_messages = []
        self.all_timestamps_ns = []

    def load(self):
        print('Load %s ...' % self.name)

        try:
            index = self.__load_index()
        except Exception as e:
            print('ERROR! Could not load topic file %s' % self.topic_filepath)
            print(str(e))
            return

        topic_info = index['topic_info']

        assert (self.name == topic_info['name'])
        self.msg_type = topic_info['msg_type']
        self.MsgClass = get_msg_class(self.msg_type)

        self.all_messages = [self.MsgClass(self.msg_type, msg_data, base_path=self.ds_dir) for msg_data in
                             index['messages']]
        self.all_timestamps_ns = [msg.get_timestamp_ns() for msg in self.all_messages]

        # assert(self.all_timestamps_ns == sorted(self.all_timestamps_ns))

    def save(self):

        write_header = False  # default is not to write header
        if os.path.isfile(self.topic_filepath):  # if the file exists, check if it is empty
            filesize = os.path.getsize(self.topic_filepath)
            if filesize == 0:
                write_header = True  # if it is empty, write header

        else:
            write_header = True  # if it does not exist, write header

        if write_header:  # if non existent or empty, no header has been added yet, so let uss add one
            out_file = open(self.topic_filepath, 'w')
            info = {'name': self.name, 'msg_type': self.msg_type}
            out_file.write('{"topic_info": %s, "messages": [\n' % json.dumps(info))

        else:  # if not empty, we need to delete the last line containing ]}, and add a comma to the end of the last line before appending the new line
            out_file = open(self.topic_filepath, 'r+')
            out_file.seek(0, 2)  # go to the end of the file
            out_file.seek(-2, os.SEEK_END)  # go to the second last character of the file

            def peek(f):
                off = f.tell()
                byte = f.read(1)
                f.seek(off, os.SEEK_SET)
                return byte

            while peek(out_file) != "\n":  # go backwards through the file to find the last "\n".
                out_file.seek(-1, os.SEEK_CUR)

            out_file.truncate()  # remove the last line
            out_file.write(',\n')  # add comma to the end of the last line before appending the new line

        # write message lines
        for msg_count, msg in enumerate(self.all_messages):
            txt = json.dumps(msg.data)

            if msg_count > 0:
                out_file.write(',\n')
            out_file.write(txt)

        # finish JSON list
        if len(self.all_messages) > 0:  # if there are messages, add a new line
            out_file.write('\n]}\n')
        else:
            out_file.write(
                ']}\n')  # if there are no messages, just write the closing bracket and newline to avoid empty lines. This can happen with tf_static
        out_file.close()

