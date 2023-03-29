from copydetect import CopyDetector

detector = CopyDetector(test_dirs = ['docs'], extensions = ['py'], display_t = 0.5)
detector.add_file("copydetect/utils.py")
detector.run()
detector.generate_html_report()