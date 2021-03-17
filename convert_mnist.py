
def convert(img_file, label_file, txt_image_file, txt_label_file, n_images):
    lbl_f = open(label_file, "rb")
    img_f = open(img_file, "rb")
    txt_lbl_f = open(txt_label_file, "w")
    txt_img_f = open(txt_image_file, "w")
    
    
    img_f.read(16)
    lbl_f.read(8)
    
    for i in range(n_images):
        lbl = ord(lbl_f.read(1))
        txt_lbl_f.write(str(lbl) + "\n")
        for j in range(784):
            val = ord(img_f.read(1))
            txt_img_f.write(str(val) + "\t")
        txt_img_f.write("\n")
    img_f.close()
    lbl_f.close()
    txt_lbl_f.close()
    txt_img_f.close()

#will take time to run
convert("./mnist/source/train-images.idx3-ubyte", "./mnist/source/train-labels.idx1-ubyte", "./mnist/train-images.idx3-ubyte.txt", "./mnist/train-labels.idx1-ubyte.txt", 60000)       
convert("./mnist/source/t10k-images.idx3-ubyte", "./mnist/source/t10k-labels.idx1-ubyte", "./mnist/t10k-images.idx3-ubyte.txt", "./mnist/t10k-labels.idx1-ubyte.txt", 10000)        