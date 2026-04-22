from icrawler.builtin import BingImageCrawler


def download(keyword, folder, max_num=100):
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=keyword, max_num=max_num)


# 🔥 APPAREL
download("t shirt mockup", "../datasets/apparel/train/apparel", 150)
download("hoodie clothing", "../datasets/apparel/train/apparel", 150)
download("t shirt on person", "../datasets/apparel/train/apparel", 150)

# 🔥 NON APPAREL
download("logo design", "../datasets/apparel/train/non_apparel", 150)
download("poster design", "../datasets/apparel/train/non_apparel", 150)
download("text meme", "../datasets/apparel/train/non_apparel", 150)