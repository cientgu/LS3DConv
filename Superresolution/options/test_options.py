from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=25, help='how many test images to run')        
        self.parser.add_argument('--use_real_img', action='store_true', help='use real image for first frame')
        self.parser.add_argument('--start_frame', type=int, default=1, help='frame index to start inference on')        
        self.parser.add_argument('--no_eval', action='store_true', help='if specified, do not eval the results')
        self.parser.add_argument('--save_image', action='store_true', help='if specified, save the output image')
        self.isTrain = False
