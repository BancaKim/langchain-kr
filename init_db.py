from sqlalchemy.orm import Session
from models import RegionGroup, RegionHeadquarter, Branch, Rank, Position
from database import engine

def init_db():
    session = Session(bind=engine)

    # Region Groups
    gangnam = RegionGroup(name="강남지역그룹")
    gyeongin = RegionGroup(name="경인지역그룹")
    session.add_all([gangnam, gyeongin])
    session.commit()

    # Region Headquarters
    gangnam1 = RegionHeadquarter(name="강남1(방배중앙)", region_group_id=gangnam.id)
    gangnam2 = RegionHeadquarter(name="강남2(신사동)", region_group_id=gangnam.id)
    gyeongin1 = RegionHeadquarter(name="경인1(용현남)", region_group_id=gyeongin.id)
    gyeongin2 = RegionHeadquarter(name="경인2(가좌공단)", region_group_id=gyeongin.id)
    session.add_all([gangnam1, gangnam2, gyeongin1, gyeongin2])
    session.commit()

    # Branches
    branches = [
        Branch(name="방배중앙종금", region_headquarter_id=gangnam1.id),
        Branch(name="방배역", region_headquarter_id=gangnam1.id),
        Branch(name="신사동종금", region_headquarter_id=gangnam2.id),
        Branch(name="신사중앙", region_headquarter_id=gangnam2.id),
        Branch(name="용현남종금", region_headquarter_id=gyeongin1.id),
        Branch(name="주안", region_headquarter_id=gyeongin1.id),
        Branch(name="가좌공단종금", region_headquarter_id=gyeongin2.id),
        Branch(name="청라", region_headquarter_id=gyeongin2.id)
    ]
    session.add_all(branches)
    session.commit()

    # Ranks
    ranks = [
        Rank(level="L0"),
        Rank(level="L1"),
        Rank(level="L2"),
        Rank(level="L3"),
        Rank(level="L4")
    ]
    session.add_all(ranks)
    session.commit()

    # Positions
    positions = [
        Position(name="계장", rank_id=ranks[0].id),
        Position(name="대리", rank_id=ranks[1].id),
        Position(name="과장", rank_id=ranks[2].id),
        Position(name="차장", rank_id=ranks[2].id),
        Position(name="부지점장", rank_id=ranks[3].id),
        Position(name="수석차장", rank_id=ranks[3].id),
        Position(name="지점장", rank_id=ranks[3].id),
        Position(name="본부장", rank_id=ranks[4].id),
        Position(name="지점장", rank_id=ranks[4].id)
    ]
    session.add_all(positions)
    session.commit()

    session.close()

if __name__ == "__main__":
    init_db()
