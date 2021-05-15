import os


def image_path_txt(image_dir, img_ext='.jpg', annot_ex='.mat'):
    """catch file name to gen
    :return:
    """
    filename_path = os.path.join(image_dir, "image.txt")
    with open(filename_path, "w") as writer:
        for root, dirs, files in os.walk(image_dir):
            for name in sorted(files):
                if img_ext in name:
                    if name[:-4] + annot_ex in files:
                        print("Write write: " + os.path.join(root.replace(image_dir+'/', ''), name[:-4]))
                        writer.write(os.path.join(root.replace(image_dir+'/', ''), name[:-4])+'\n')
                else:
                    continue
    writer.close()
    return


if __name__ == '__main__':
    img_dir = "/data/deep-head-pose/300W_LP"  # revise for your dataset path
    image_path_txt(img_dir)
