from sqlalchemy import Table, Index, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
 
Base = declarative_base()
 
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False)
    screen_name = Column(String(250), nullable=False)
 
tags_tweets = Table('tags_tweets', Base.metadata,
    Column('tweet_id', Integer, ForeignKey('tweet.id')),
    Column('tag_id', Integer, ForeignKey('tag.id')),
    Index('idx_tweet_tag', 'tweet_id', 'tag_id', unique=True)
)

class Tweet(Base):
    __tablename__ = 'tweet'
    id = Column(Integer, primary_key=True)
    tweet_text = Column(String(250))
    timestamp = Column(DateTime())
    picture_url = Column(String(250))
    user_id = Column(Integer, ForeignKey('user.id'))
    user = relationship(User)
    tags = relationship("Tag",
                    secondary=tags_tweets)

class Tag(Base):
    __tablename__ = 'tag'
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False, unique=True)
    tweets = relationship("Tweet",
                    secondary=tags_tweets)

# This will be run only when we execute file from shell,
# e.g. 'python create_schema.py'; when imported in another file
# only methods are imported

def main():
  # Create an engine that stores data in the local directory's
  # sqlalchemy_example.db file.
  engine = create_engine('sqlite:///tweets.db')
 
  # Create all tables in the engine. This is equivalent to "Create Table"
  # statements in raw SQL.
  Base.metadata.create_all(engine)

if __name__ == '__main__':
   main()
