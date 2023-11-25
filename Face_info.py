python


class FaceInfo:
    def __init__(self, input_type, image_path):
        self.input_type = input_type
        self.image_path = image_path

    def process_image(self):
        frame = cv2.imread(self.image_path)
        out = f_Face_info.get_face_info(frame)
        res_img = f_Face_info.bounding_box(out, frame)
        cv2.imshow('Face info', res_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Info")
    parser.add_argument('--input', type=str, default='webcam', help="webcam or image")
    parser.add_argument('--path_im', type=str, help="path of image")
    args = vars(parser.parse_args())

    face_info = FaceInfo(args['input'], args['path_im'])
    face_info.process_image()
