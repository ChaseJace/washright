from icrawler.builtin import GoogleImageCrawler

# Create a Google crawler that downloads images into a custom folder
google_crawler = GoogleImageCrawler(storage={'root_dir': 'stain_dataset/cookingoil_stains2'})

# Start crawling with a specific search keyword
google_crawler.crawl(keyword='cooking oil stain on clothes', max_num=200)
