import argparse
# import cv2
import sys
import numpy as np
import os
import face_model
from scipy import misc
# import rm_dim

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--data_path', type=str,default='/data/liukang/face_data/toukui_testdata_size112/ID_ID218_align160/', help='data path')
    parser.add_argument('--output_dir', type=str,default=os.path.join(os.path.dirname(__file__), '..','Saveout'), help='data path')
    parser.add_argument('--image_size', default='112,112', help='')
    parser.add_argument('--model', default='/data/fengchen/ensemble/model/model-r100-ii/model,0', help='path to load model.')
    parser.add_argument('--ga_model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=1, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='mtcnn opttion,ver dist threshold')
    parser.add_argument('--dim_threshold', default=30, type=float, help='mtcnn opttion,ver dist threshold')
    return parser.parse_args(argv)

def gen_feature(args):
    #loading model
    model = face_model.FaceModel(args)
    no_get_feature=0
    rm_dim_number=0
    #get feature
    feature_lebal_id={"feature":[],"lebal":[],"id":[]}
    output_dir=args.output_dir
    if not  os.path.exists (output_dir):
        os.makedirs(output_dir)
    lebal_list=os.listdir(args.data_path)
    print("class number",len(lebal_list))
    for i_id in range(len(lebal_list)):
        lebal_path=os.path.join(args.data_path,lebal_list[i_id])
        if os.path.isdir(lebal_path):
            print("procesing:{}/{}".format(i_id ,len(lebal_list)))
            for img in os.listdir(lebal_path):
                img_path=os.path.join(lebal_path,img)
                # print('img_path',img_path)
                try:
                    # img_RGB = cv2.imread(img_path)
                    # dim=rm_dim.rm_main(img_RGB,args.dim_threshold)
                    # if dim==1:
                    #     rm_dim_number=rm_dim_number+1
                    #     continue
                    # img_input = model.get_input(img_RGB)
                    img_rgb = misc.imread(img_path)
                    img_input = np.transpose(img_rgb, (2, 0, 1))
                    feature= model.get_feature(img_input)
                    feature_lebal_id['feature'].append(feature)
                    feature_lebal_id['lebal'].append(lebal_list[i_id])
                    feature_lebal_id['id'].append(i_id)
                except (Exception):
                    no_get_feature=no_get_feature+1
                    print("get feature fail",img_path)
                    continue
    
    
    print('dim_images_number',rm_dim_number)
    print('fail to get feature',no_get_feature)
    print('success to get lebal number',len(feature_lebal_id['lebal']),len(feature_lebal_id['feature']))
    np.save(os.path.join(output_dir, "labels_name.npy"), feature_lebal_id['lebal'])
    np.save(os.path.join(output_dir, "gallery.npy"), feature_lebal_id['id'])
    np.save(os.path.join(output_dir, "signatures.npy"), feature_lebal_id['feature'])

if __name__ == '__main__':
    gen_feature(parse_arguments(sys.argv[1:]))
