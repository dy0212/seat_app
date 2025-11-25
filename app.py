from typing import List
from dataclasses import dataclass
from flask import Flask, render_template, request

INF = 10**18

# =========================
# ⚙️ 0. 조절 가능한 파라미터 (여기만 고치면 됨)
# =========================

NUM_ROWS = 3
NUM_COLS = 4
NUM_STUDENTS = 6
NUM_SEATS = NUM_ROWS * NUM_COLS

# 학생별 좌석 호감도: STUDENT_SEAT_SCORES[학생][seat_id]
STUDENT_SEAT_SCORES: List[List[int]] = [
    [10,  9,  8,  7,
      6,  5,  4,  3,
      2,  1,  0, -1],

    [ 3,  4,  5,  6,
      7,  8,  9, 10,
      5,  6,  7, 11],

    [ 5,  6,  7,  6,
      8,  9, 10,  8,
      6,  5,  4,  3],

    [ 7,  8,  9, 10,
      5,  6,  7,  8,
      3,  4,  5,  6],

    [ 5,  5,  5,  5,
      5,  5,  5,  5,
      5,  5,  5,  5],

    [10,  4,  4,  4,
     10,  4,  4,  4,
     10,  4,  4,  4],
]
assert len(STUDENT_SEAT_SCORES) == NUM_STUDENTS
assert all(len(row) == NUM_SEATS for row in STUDENT_SEAT_SCORES)

# 친구/적 호감도 (대칭 + 대각선 0)
FRIEND: List[List[int]] = [
    [ 0,  5, -3,  0,  0,  0],
    [ 5,  0,  2,  0,  0,  0],
    [-3,  2,  0,  4,  0,  0],
    [ 0,  0,  4,  0,  1, -2],
    [ 0,  0,  0,  1,  0,  0],
    [ 0,  0,  0, -2,  0,  0],
]
assert len(FRIEND) == NUM_STUDENTS and all(len(row) == NUM_STUDENTS for row in FRIEND)

PROX_RADIUS = 3  # 맨해튼 거리 0~2까지만 영향

STUDENT_NAMES = [f"S{i+1}" for i in range(NUM_STUDENTS)]  # 웹에서 표시할 이름


# =========================
# 1. 헝가리안 알고리즘
# =========================

def hungarian(cost: List[List[int]]) -> List[int]:
    n = len(cost)
    m = len(cost[0])
    assert n == m, "N x N 정사각 행렬이어야 합니다."

    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def max_satisfaction_assignment(scores: List[List[int]]) -> List[int]:
    n = len(scores)
    m = len(scores[0])
    assert n == m

    max_score = max(scores[i][j] for i in range(n) for j in range(m))
    cost = [[max_score - scores[i][j] for j in range(m)] for i in range(n)]
    return hungarian(cost)


# =========================
# 2. 좌석/거리 모델링
# =========================

@dataclass
class Seat:
    id: int
    row: int
    col: int


def create_seats(num_rows: int, num_cols: int) -> List[Seat]:
    seats: List[Seat] = []
    seat_id = 0
    for r in range(num_rows):
        for c in range(num_cols):
            seats.append(Seat(id=seat_id, row=r, col=c))
            seat_id += 1
    return seats


def manhattan_dist(a: Seat, b: Seat) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


# =========================
# 3. 점수 행렬 생성 + 총점 계산
# =========================

def build_score_matrix(num_students: int,
                       seats: List[Seat],
                       student_seat_scores: List[List[int]]) -> List[List[int]]:
    used_seats = seats[:num_students]
    scores: List[List[int]] = []
    for i in range(num_students):
        row: List[int] = []
        for seat in used_seats:
            seat_id = seat.id
            row.append(student_seat_scores[i][seat_id])
        scores.append(row)
    return scores


def total_score(assignment: List[int],
                seats: List[Seat],
                student_seat_scores: List[List[int]],
                friend: List[List[int]],
                prox_radius: int) -> int:
    n = len(assignment)
    used_seats = seats[:n]

    seat_part = 0
    for i in range(n):
        seat_idx = assignment[i]
        seat_id = used_seats[seat_idx].id
        seat_part += student_seat_scores[i][seat_id]

    friend_part = 0
    for i in range(n):
        for j in range(i + 1, n):
            aff = friend[i][j]
            if aff == 0:
                continue
            s_i = used_seats[assignment[i]]
            s_j = used_seats[assignment[j]]
            d = manhattan_dist(s_i, s_j)
            closeness = max(0, prox_radius - d)
            friend_part += aff * closeness

    return seat_part + friend_part


def improve_with_friendship(assignment: List[int],
                            seats: List[Seat],
                            student_seat_scores: List[List[int]],
                            friend: List[List[int]],
                            prox_radius: int) -> (List[int], int):
    n = len(assignment)
    best = total_score(assignment, seats, student_seat_scores, friend, prox_radius)
    improved = True

    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                assignment[i], assignment[j] = assignment[j], assignment[i]
                new_score = total_score(assignment, seats, student_seat_scores, friend, prox_radius)
                if new_score > best:
                    best = new_score
                    improved = True
                else:
                    assignment[i], assignment[j] = assignment[j], assignment[i]
            if improved:
                break

    return assignment, best


# =========================
# 4. Flask 앱
# =========================

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    # 행/열/학생 수를 폼으로 바꾸고 싶으면 여기를 request.form에서 읽게 변경하면 됨.
    seats = create_seats(NUM_ROWS, NUM_COLS)

    score_matrix = build_score_matrix(NUM_STUDENTS, seats, STUDENT_SEAT_SCORES)
    assignment = max_satisfaction_assignment(score_matrix)
    assignment, best_score = improve_with_friendship(
        assignment, seats, STUDENT_SEAT_SCORES, FRIEND, PROX_RADIUS
    )

    # seat_grid[row][col] = 학생 이름 or None
    seat_grid: List[List[str | None]] = [
        [None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)
    ]

    # seat_index -> student index
    seat_to_student = {assignment[i]: i for i in range(NUM_STUDENTS)}

    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            seat_id = r * NUM_COLS + c
            if seat_id >= NUM_STUDENTS:
                continue
            seat_index = seat_id  # seats[:N] -> id == index
            stu_idx = seat_to_student.get(seat_index)
            if stu_idx is not None:
                seat_grid[r][c] = STUDENT_NAMES[stu_idx]

    return render_template(
        "index.html",
        rows=NUM_ROWS,
        cols=NUM_COLS,
        seat_grid=seat_grid,
        best_score=best_score,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

