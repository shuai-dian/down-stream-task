from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="./03.png",help='Image file')
    parser.add_argument('--config',default="../local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py", help='Config file')
    parser.add_argument('--checkpoint',default="../weights/segformer.b5.640x640.ade.160k.pth",help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    print(result)

    # show the results
    show_result_pyplot(model, args.img, result, get_palette(args.palette))


if __name__ == '__main__':
    main()
