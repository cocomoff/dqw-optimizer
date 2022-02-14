import pulp
import json
import pickle
import numpy as np
from typing import List, Tuple, Dict
from enum import IntFlag, auto
from pprint import pprint


# # 心の色
# class KokoroColor(IntEnum):
#     """
#     赤 黄 青 紫 緑
#     """
#     Red = auto()
#     Yellow = auto()
#     Blue = auto()
#     Purple = auto()
#     Green = auto()


# 心・枠の色
class Color(IntFlag):
    Red = auto()
    Yellow = auto()
    Blue = auto()
    Purple = auto()
    Green = auto()
    RedYellow = Red | Yellow  # バトルマスター(2)
    RedBlue = Red | Blue  # レンジャー(2)
    PurpleGreen = Purple | Green  # 賢者(2,3,4)
    YellowPurple = Yellow | Purple  # 魔法戦士(2,3,4)
    YellowGreen = Yellow | Green  # パラディン(2)
    YellowBlue = Yellow | Blue  # 海賊(2)
    BlueGreen = Blue | Green  # スーパースター(2)
    Rainbow = Red | Yellow | Blue | Purple | Green  # 虹枠

    def __str__(self):
        str_rep = "!!!"
        if self is Color.Red:
            str_rep = "赤"
        elif self is Color.Yellow:
            str_rep = "黄"
        elif self is Color.Blue:
            str_rep = "青"
        elif self is Color.Purple:
            str_rep = "紫"
        elif self is Color.Green:
            str_rep = "緑"
        elif self is Color.RedYellow:
            str_rep = "赤|黄"
        elif self is Color.RedBlue:
            str_rep = "赤|青"
        elif self is Color.PurpleGreen:
            str_rep = "紫|緑"
        elif self is Color.YellowPurple:
            str_rep = "黄|紫"
        elif self is Color.YellowGreen:
            str_rep = "黄|緑"
        elif self is Color.YellowBlue:
            str_rep = "黄|青"
        elif self is Color.BlueGreen:
            str_rep = "青|緑"
        elif self is Color.Rainbow:
            str_rep = "赤|黄|青|紫|緑"
        else:
            pass
        return f"[{str_rep}]"


# 枠
roles: List[str] = ["バトルマスター", "賢者", "レンジャー", "魔法戦士", "パラディン", "スーパースター", "海賊"]

# 職業ごとの枠
waku_dict = {
    "バトルマスター": [Color.Rainbow, Color.RedYellow, Color.Red, Color.Red],
    "賢者": [
        Color.Rainbow,
        Color.PurpleGreen,
        Color.PurpleGreen,
        Color.PurpleGreen,
    ],
    "レンジャー": [Color.Rainbow, Color.RedBlue, Color.Blue, Color.Blue],
    "魔法戦士": [
        Color.Rainbow,
        Color.YellowPurple,
        Color.YellowPurple,
        Color.YellowPurple,
    ],
    "パラディン": [
        Color.Rainbow,
        Color.YellowGreen,
        Color.Yellow,
        Color.Yellow,
    ],
    "スーパースター": [
        Color.Rainbow,
        Color.BlueGreen,
        Color.Blue,
        Color.Green,
    ],
    "海賊": [Color.Rainbow, Color.YellowBlue, Color.Yellow, Color.Blue],
}


# パラメータ
parameters: List[str] = [
    "さいだいHP",
    "さいだいMP",
    "ちから",
    "みのまもり",
    "こうげき魔力",
    "かいふく魔力",
    "すばやさ",
    "きようさ",
]

ranks: List[str] = [
    "Sランク",
    "Aランク",
    "Bランク",
    "Cランク",
    "Dランク",
]

# パラメータと補正; color x parameter size
color2index: Dict[Color, int] = {
    Color.Yellow: 0,
    Color.Green: 1,
    Color.Red: 2,
    Color.Purple: 3,
    Color.Blue: 4,
}

colorname2color: Dict[str, Color] = {
    "黄": Color.Yellow,
    "緑": Color.Green,
    "赤": Color.Red,
    "紫": Color.Purple,
    "青": Color.Blue,
}

coeff: np.ndarray = np.array(
    [
        # 黄   緑   赤   紫   青
        [1.2, 1.0, 1.0, 1.0, 1.2],  # さいだいHP
        [1.0, 1.2, 1.0, 1.2, 1.0],  # さいだいMP
        [1.2, 1.0, 1.2, 1.0, 1.0],  # ちから
        [1.2, 1.0, 1.0, 1.0, 1.0],  # みのまもり
        [1.0, 1.0, 1.0, 1.2, 1.0],  # こうげき魔力
        [1.0, 1.2, 1.0, 1.0, 1.0],  # かいふく魔力
        [1.0, 1.0, 1.2, 1.0, 1.2],  # すばやさ
        [1.0, 1.0, 1.2, 1.0, 1.2],  # きようさ
    ]
)


def naive_optimizer(
    syokugyou: str = "バトルマスター", max_cost: int = 308
) -> Tuple[int, str, str]:
    """
    最適化して返す (例; バトルマスター Lv.55 max_cost 308)
    """
    # read data
    monster_dict = pickle.load(open("notebook/dqw.pickle", "rb"))

    problem = pulp.LpProblem(name="DQW kokoroptimizer", sense=pulp.LpMaximize)

    # 決定変数は (id, rank) のキー (rankは0,1,2,3,4=S,A,B,C,D)
    valid_ids = []
    dict_discount = {}
    dict_cost = {}
    for key in monster_dict:
        """
        例外処理 (もみじこぞう) 色ミス
        """
        if "type" not in monster_dict[key]:
            continue
        mid = monster_dict[key]["id"]
        name = monster_dict[key]["name"]
        cost = monster_dict[key]["cost"]
        color = monster_dict[key]["type"]
        valid_ids.append(key)
        dict_discount[key] = {}
        dict_cost[key] = cost

        print(mid, name, cost, color)
        # pprint(monster_dict[key])

        for name in monster_dict[key]["special"]:
            rank = ranks.index(name)
            # print(monster_dict[key]["special"][name])
            for elem in monster_dict[key]["special"][name]:
                if elem.startswith("こころ最大コスト"):
                    target = elem.split("+")[-1]
                    # 表記ミス%
                    if target.endswith("%"):
                        target = target[:-1]
                    # 表記ミス+忘れ
                    if target == "こころ最大コスト4":
                        target = "4"
                    num = int(target)
                    dict_discount[key][rank] = num

    var = pulp.LpVariable.dicts("x", (valid_ids, range(5)), 0, 1, "Integer")

    # 制約 (排他)
    for key in valid_ids:
        problem += pulp.lpSum(var[key][rank] for rank in range(5)) <= 1

    # 制約 (コスト)
    term = 0
    for key in valid_ids:
        term += pulp.lpSum(
            (
                dict_cost[key] - dict_discount[key][rank]
                if rank in dict_discount[key]
                else dict_cost[key]
            )
            * var[key][rank]
            for rank in range(5)
        )
    problem += term <= max_cost

    # 制約: 4つまで
    problem += pulp.lpSum(var[key][rank] for rank in range(5) for key in valid_ids) <= 4

    # 目的関数: 最大パラメータ値
    obj = 0
    for key in valid_ids:
        print(key)
        terms = []
        for param in parameters:
            term = pulp.lpSum(
                monster_dict[key][param][rank] * var[key][rank] for rank in range(5)
            )
            obj += term

    # debug output
    problem += obj

    # solve
    problem.solve()
    # print(pulp.value(problem.objective))
    result = []
    for key in valid_ids:
        v = [pulp.value(var[key][rank]) for rank in range(5)]
        if sum(v) >= 1:
            rank = ""
            for r in range(5):
                if v[r] > 0.0:
                    rank = ranks[r]
                    break
            result.append((key, monster_dict[key]["name"], rank))

    return_result = {}
    for i in range(4):
        return_result[i] = result[i]
    return return_result


def color_based_optimizer(syokugyou: str = "バトルマスター", max_cost: int = 308) -> None:
    """
    最適化して返す (例; バトルマスター Lv.55 max_cost 308)
    """
    # read data
    monster_dict = pickle.load(open("notebook/dqw.pickle", "rb"))

    problem = pulp.LpProblem(
        name="DQW kokoroptimizer with color", sense=pulp.LpMaximize
    )

    print("-----waku-----")
    for waku in waku_dict[syokugyou]:
        print(waku)
    print("--------------")
    print()

    id_to_pair = {}
    param_dict = {}
    for key in monster_dict:
        d = monster_dict[key]

        # mid = d["id"]
        # name = d["name"]
        # cost = d["cost"]
        # color = d["type"]

        for rank_name in ranks:
            id_to_pair[(key, rank_name)] = len(id_to_pair)

        param_dict[key] = {}
        for (idr, rank_name) in enumerate(ranks):
            values = [d[param][idr] for param in parameters]
            param_dict[key][rank_name] = values

    # 決定変数
    # 4つの枠 (各color) に対してどの (num, rank) を割り当てるのか
    # (num [monster_id], rank [0=S, 1=A, 2=B, 3=C, 4=D])
    vars = pulp.LpVariable.dicts("X", range(4), 0, len(monster_dict) * 5, "Integer")
    print(vars)

    # valid_ids = []
    # dict_discount = {}
    # dict_cost = {}
    # for key in monster_dict:
    #     mid = monster_dict[key]["id"]
    #     name = monster_dict[key]["name"]
    #     cost = monster_dict[key]["cost"]
    #     color = monster_dict[key]["type"]

    # # 決定変数は (id, rank) のキー (rankは0,1,2,3,4=S,A,B,C,D)
    # valid_ids = []
    # dict_discount = {}
    # dict_cost = {}
    # for key in monster_dict:
    #     mid = monster_dict[key]["id"]
    #     name = monster_dict[key]["name"]
    #     cost = monster_dict[key]["cost"]
    #     color = monster_dict[key]["type"]
    #     valid_ids.append(key)
    #     dict_discount[key] = {}
    #     dict_cost[key] = cost

    #     # print(mid, name, cost, color)
    #     # pprint(monster_dict[key])

    #     for name in monster_dict[key]["special"]:
    #         rank = ranks.index(name)
    #         # print(monster_dict[key]["special"][name])
    #         for elem in monster_dict[key]["special"][name]:
    #             if elem.startswith("こころ最大コスト"):
    #                 target = elem.split("+")[-1]
    #                 # 表記ミス%
    #                 if target.endswith("%"):
    #                     target = target[:-1]
    #                 # 表記ミス+忘れ
    #                 if target == "こころ最大コスト4":
    #                     target = "4"
    #                 num = int(target)
    #                 dict_discount[key][rank] = num

    # var = pulp.LpVariable.dicts("x", (valid_ids, range(5)), 0, 1, "Integer")

    # # 制約 (排他)
    # for key in valid_ids:
    #     problem += pulp.lpSum(var[key][rank] for rank in range(5)) <= 1

    # # 制約 (コスト)
    # term = 0
    # for key in valid_ids:
    #     term += pulp.lpSum(
    #         (
    #             dict_cost[key] - dict_discount[key][rank]
    #             if rank in dict_discount[key]
    #             else dict_cost[key]
    #         )
    #         * var[key][rank]
    #         for rank in range(5)
    #     )
    # problem += term <= max_cost

    # # 制約: 4つまで
    # problem += pulp.lpSum(var[key][rank] for rank in range(5) for key in valid_ids) <= 4

    # # 目的関数: 最大パラメータ値
    # obj = 0
    # for key in valid_ids:
    #     terms = []
    #     for param in parameters:
    #         term = pulp.lpSum(
    #             monster_dict[key][param][rank] * var[key][rank] for rank in range(5)
    #         )
    #         obj += term

    # # debug output
    # problem += obj

    # # solve
    # problem.solve(pulp.PULP_CBC_CMD(msg=0))
    # # print(pulp.value(problem.objective))
    # result = []
    # for key in valid_ids:
    #     v = [pulp.value(var[key][rank]) for rank in range(5)]
    #     if sum(v) >= 1:
    #         rank = ""
    #         for r in range(5):
    #             if v[r] > 0.0:
    #                 rank = ranks[r]
    #                 break
    #         result.append((key, monster_dict[key]["name"], rank))

    # return_result = {}
    # for i in range(4):
    #     return_result[i] = result[i]
    return None


list_objectives: List[str] = ["最大パラメータ総和値", "魔法重視", "回復重視"]

weight_objs: np.ndarray = np.array(
    [
        # max  魔  回
        [1.0, 1.0, 1.0],  # さいだいHP
        [1.0, 2.0, 2.0],  # さいだいMP
        [1.0, 0.0, 0.0],  # ちから
        [1.0, 0.0, 0.0],  # みのまもり
        [1.0, 3.0, 1.0],  # こうげき魔力
        [1.0, 1.0, 3.0],  # かいふく魔力
        [1.0, 0.0, 0.0],  # すばやさ
        [1.0, 0.0, 0.0],  # きようさ
    ]
)


def new_optimizer(
    syokugyou: str = "バトルマスター",
    objective: str = "最大パラメータ総和値",
    max_cost: int = 308,
    num_kokoro: int = 4,
    constraints: set = set({}),
) -> Tuple[int, str, str]:
    """
    最適化して返す (例; バトルマスター Lv.55 max_cost 308)
    """
    # read data
    monster_dict = json.load(open("notebook/dqw.json", "r"))
    problem = pulp.LpProblem(name="DQW-kokoroptimizer", sense=pulp.LpMaximize)

    assert syokugyou in roles
    assert objective in list_objectives

    # 決定変数は (id, rank) のキー (rankは0,1,2,3,4=S,A,B,C,D)
    ids = list(monster_dict.keys())
    dict_discount = {}
    dict_cost = {}
    for key in monster_dict:
        mid = monster_dict[key]["id"]
        name = monster_dict[key]["name"]
        cost = monster_dict[key]["cost"]
        dict_discount[key] = {}
        dict_cost[key] = cost
        for name in monster_dict[key]["special"]:
            rank = ranks.index(name)
            for elem in monster_dict[key]["special"][name]:
                if elem.startswith("こころ最大コスト"):
                    target = elem.split("+")[-1]
                    num = int(target)
                    dict_discount[key][rank] = num

    # |ids| x |rank(S,A,B,C,D)| x |index(0,1,2,3)|
    # 0 [使わない] 1-4 (1つ目～4つ目に使う)
    rng_r = range(len(ranks))
    rng_k = range(num_kokoro)
    X = pulp.LpVariable.dicts("x", (ids, rng_r, rng_k), 0, 1, "Integer")

    # 制約 | 各モンスターは1回しか使えない
    for mid in ids:
        problem += pulp.lpSum(X[mid][rank][loc] for rank in rng_r for loc in rng_k) <= 1

    # 4つしか使えない
    problem += (
        pulp.lpSum(X[mid][rank][loc] for mid in ids for rank in rng_r for loc in rng_k)
        <= num_kokoro
    )

    # 各位置(loc)は全て1度使う
    for loc in rng_k:
        problem += pulp.lpSum(X[mid][rank][loc] for mid in ids for rank in rng_r) == 1

    # 制約 (コスト)
    cost_term = 0
    for key in ids:
        cost_term += pulp.lpSum(
            (
                dict_cost[key] - dict_discount[key][rank]
                if rank in dict_discount[key]
                else dict_cost[key]
            )
            * X[key][rank][loc]
            for rank in rng_r
            for loc in rng_k
        )
    problem += cost_term <= max_cost

    # 目的関数: 最大パラメータ値
    # 各位置について，色補正付きの項を計算する
    obj = 0.0
    W = weight_objs[:, list_objectives.index(objective)]
    for loc in rng_k:
        color_loc = waku_dict[syokugyou][loc]
        # loc位置のパラメータごとの目的関数寄与分
        list_obj_loc = []
        for (idp, param) in enumerate(parameters):
            term_param = 0.0
            for mid in ids:
                mtype = colorname2color[monster_dict[mid]["type"]]
                value_coeff = 1.0
                if mtype & color_loc == mtype:
                    value_coeff = coeff[idp, color2index[mtype]]
                for rank in rng_r:
                    term_param += (
                        monster_dict[mid][param][rank] * value_coeff * X[mid][rank][loc]
                    )
            list_obj_loc.append(term_param)

        for (idp, _) in enumerate(parameters):
            obj += W[idp] * list_obj_loc[idp]

    # ないやつ
    # target = set({("ワイトキング", 0)})
    for mid in ids:
        name = monster_dict[mid]["name"]
        for rank in rng_r:
            if (name, rank) in constraints:  # target:
                for loc in rng_k:
                    problem += X[mid][0][loc] == 0

    # debug output
    problem += obj

    # solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    result = {}
    for key in ids:
        name = monster_dict[key]["name"]
        for rank in rng_r:
            for loc in rng_k:
                if pulp.value(X[key][rank][loc]) >= 0.5:
                    result[loc] = (
                        f"{int(key):>03d}{name}",
                        monster_dict[key]["cost"],
                        ranks[rank],
                        (name, rank),
                    )

    total_cost = 0
    # print(f"given constraints: {constraints}")
    for i in range(len(result)):
        ci = waku_dict[syokugyou][i]
        print(f"こころ{i+1} {result[i][0]} (コスト{result[i][1]}) ({result[i][2]}) ({ci})")
        total_cost += result[i][1]
    print(f"コスト {total_cost}/{max_cost}")
    print()
    return result


if __name__ == "__main__":
    # res = color_based_optimizer()
    # for k in range(4):
    #     print(res[k])
    # new_optimizer(syokugyou="スーパースター", max_cost=406)
    new_optimizer(syokugyou="魔法戦士", max_cost=440)
    new_optimizer(syokugyou="魔法戦士", objective="魔法重視", max_cost=440)
    new_optimizer(syokugyou="魔法戦士", objective="回復重視", max_cost=440)

    new_optimizer(syokugyou="パラディン", max_cost=440)
    new_optimizer(syokugyou="パラディン", objective="魔法重視", max_cost=440)
    new_optimizer(syokugyou="パラディン", objective="回復重視", max_cost=440)
