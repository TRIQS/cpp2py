# Acts as cp -rs on Linux
# Workaround for the absence of this option on OS X
# Usage cp_rs SRC DEST
# It makes in DEST the subdirectory structure of SRC and links all files with symbolic links.

import os, sys
 
rootDir, destDir = sys.argv[1:3]
extension_kept = sys.argv[3].split()

def ensuredirs(d):
    if not os.path.exists(d): 
        os.makedirs(d)

ensuredirs(destDir)
os.chdir(destDir)

for dirName, ignored_subdirList, fileList in os.walk(rootDir, followlinks = False):
    d = os.path.relpath(dirName, rootDir)
    ensuredirs(d)
    cwd = os.getcwd()
    os.chdir(d)
    for fname in fileList:
        if fname.startswith('CMakeLists'): continue
        ext = os.path.splitext(fname)[1]
        if ext[1:] not in extension_kept : continue
        if not os.path.exists(fname):
            os.symlink(os.path.join(rootDir, d, fname), fname)
    os.chdir(cwd)

