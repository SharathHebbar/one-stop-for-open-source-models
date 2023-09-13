import os


def remove_files(dirs):
    for f in os.listdir(dirs):
        os.remove(os.path.join(dirs, f))


def make_directory_if_not_exists(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        remove_files(dir_name)

