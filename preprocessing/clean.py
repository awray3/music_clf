"""
The purpose of this script is to clean the mp3 directory of
damaged mp3 files that don't load.
"""
import os
import torchaudio


def clean_mp3_directory(audio_dir):
    """
    module for checking corruptness of mp3s. Loops over the given
    directory attempting to load info on the tracks. If loading fails,
    the user will be asked if they want to remove the corrupt files
    from their system. If a file is removed, then the corresponding row from the
    meta csv will also be removed and returned by this method.
    """
    corrupted_list = []
    for root, dirs, files in os.walk(audio_dir):
        if dirs == []:
            for f in files:
                try:
                    print('Loading ' + f, end='\r')
                    torchaudio.info(os.path.join(root, f))
                except RuntimeError as e:
                    print("Unable to load file" + f + ", " + repr(e))
                    corrupted_list.append(os.path.join(root, f))

    if len(corrupted_list) == 0:
        print('No mp3s were found to be corrupted. Exiting cleaning.')

    else:
        print('The following files were found to be corrupted:')
        for fpath in corrupted_list:
            print(fpath)
        delete_option = input(
            'Would you like to delete these files? (yes/no): ')
        if delete_option == 'yes':
            # delete the files
            for fpath in corrupted_list:
                os.remove(fpath)
                print('Removed ' + os.path.split(fpath)[1][:-4] + '.')

        corrupted_list = [int(os.path.split(fpath)[1][:-4]) for fpath in corrupted_list]

    print('Exiting cleaning.')

    return corrupted_list
