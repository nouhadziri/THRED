import os

from thred.util import fs


class RedditBotHandler:

    def __init__(self):
        super(RedditBotHandler, self).__init__()

        self.bots = set()
        self.bots.add('[deleted]')
        with open(self.__get_bot_file(), 'r') as file:
            for username in file:
                if not username.startswith("#"):
                    self.bots.add(username.strip().lower())

    def __get_bot_file(self):
        return os.path.join(fs.get_current_dir(__file__), 'reddit_bots.txt')

    def is_bot(self, username):
        return username.lower() in self.bots
