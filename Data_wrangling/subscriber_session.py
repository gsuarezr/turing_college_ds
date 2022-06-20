from unicodedata import name
from subscriber_app import follow, posts, post_message, user_posts, followers, following, post_by_user, post_for_user, search
from pprint import pprint

post_message('guido', 'Me gustan las waifu japonesas')
post_message('roxana', 'Quiero el nuevo Iphone')
post_message('degen', 'El bear market me dejo en quiebra')
post_message('davin', 'Mis plantas estan creciendo rapido')
post_message('guido', 'Me gustan las waifu japonesas parte 2')
post_message('guido', 'Me gustan las waifu japonesas parte 3')
follow('guido', followed_user='degen')
follow('roxana', followed_user='degen')

if __name__ == '__main__':
   # pprint(posts)
    # pprint(user_posts['degen'])
    # pprint(following)
    # pprint(followers)
    # pprint(post_by_user('guido', limit=2))
    # pprint(post_for_user('roxana'))
    pprint(search('waifu', 2))
