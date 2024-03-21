import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/basmati', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

    # (Jakob) Q1.1 do scene and book
    q1_1_chosen_ratio_threshold = 0.7
    q1_2_new_ratio_threshold = 0.9
    get_1_correspondance_with_book = 0.249
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/book', ratio_thres=q1_1_chosen_ratio_threshold)
    plt.title('Match')
    plt.imshow(im)

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/scene', './data/basmati',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    
    # (Jakob) Q1.2 do scene and book with Ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/scene', './data/book',
        ratio_thres=q1_1_chosen_ratio_threshold, orient_agreement=30, scale_agreement=0.5)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    
    # poor library without ransac, but same threshol
    mon_ratio = 0.8
    mon_orient_agree = 25
    mon_scale_agree = 0.1
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/library', './data/library2', ratio_thres=mon_ratio)
    plt.title('Match')
    plt.imshow(im)
    
    # ransac library, pre good.
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=mon_ratio, orient_agreement=mon_orient_agree, scale_agreement=mon_scale_agree)
    plt.title('MatchRANSAC')
    plt.imshow(im)
    
    # decent library without ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/library', './data/library2', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

if __name__ == '__main__':
    main()
