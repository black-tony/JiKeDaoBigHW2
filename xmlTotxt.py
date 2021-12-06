
import os
import xml.etree.ElementTree as ET
import glob
import json

def xml_to_txt(indir,outdir):

    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')

    file_save = 'all.txt'
    file_txt=os.path.join(outdir,file_save)
    f_w = open(file_txt,'w')

    for i, file in enumerate(annotations):

        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()

        filename = root.find('filename').text

        f_w.write(filename+'    ')

        for obj in root.iter('object'):

            dir = {}

            name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xn = float(xmlbox.find('xmin').text)
            xx = float(xmlbox.find('xmax').text)
            yn = float(xmlbox.find('ymin').text)
            yx = float(xmlbox.find('ymax').text)
            
            dir["value"] = name
            dir["coordinate"] = [[xn, yn], [xx, yx]]
            f_w.write(json.dumps(dir))
            f_w.write(' ')
        
        f_w.write('\n')

indir='./'   #xml目录
outdir='./'  #txt目录

xml_to_txt(indir,outdir)