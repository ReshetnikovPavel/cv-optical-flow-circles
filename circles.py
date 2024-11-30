import random

import numpy as np
import cv2


class Circle:
    def __init__(self, x, y, r, color):
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.force = np.array([0, 0])


class State:
    def __init__(self, screen_width, screen_height, circles_count=15):
        self.scale_factor = 0.2
        self.scaled_width = round(screen_width * self.scale_factor)
        self.scaled_height = round(screen_height * self.scale_factor)
        self.prev_frame = None
        self.flow = None

        # В самом начале создаю `circles_count` кружочков в случайном месте со случайным размером и цветом
        self.circles = [
            random_circle(screen_width, screen_height) for _ in range(circles_count)
        ]

    def process(self, frame):
        # Отражаю изображение, т.к. так удобнее, когда экран работает как зеркало
        frame = cv2.flip(frame, 1)

        # Уменьшаю для производительности и делаю серым
        frame_gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (self.scaled_width, self.scaled_height),
            interpolation=cv2.INTER_AREA,
        )

        if self.prev_frame is not None:
            self.flow = optical_flow(frame_gray, self.prev_frame, self.flow)
            self.move_circles(frame, self.flow)
            frame = self.draw_circles(frame)

        self.prev_frame = frame_gray
        return frame

    def draw_circles(self, frame):
        for circle in self.circles:
            frame = cv2.circle(frame, (circle.x, circle.y),
                               circle.r, circle.color, -1)
        return frame

    def move_circles(self, frame, flow):
        for circle in self.circles:
            # Создаю маску, где на кадре в уменьшенном разрешении находится этот кружочек
            circle_mask = cv2.circle(
                np.zeros((self.scaled_width, self.scaled_height),
                         dtype=np.uint8),
                (
                    round(circle.x * self.scale_factor),
                    round(circle.y * self.scale_factor),
                ),
                round(circle.r * self.scale_factor),
                255,
                -1,
            )
            # применяю маску
            circle_flow = flow[circle_mask != 0]

            # Складываю все вектора под маской и беру с обратным знаком, чтобы шарики двигались в нужную сторону
            forces = -np.add.reduce(circle_flow)

            # Создаю некоторое трение, чтобы шарики не двигались вечно сами по себе
            circle.force = 0.97 * circle.force
            circle.force += forces

            # Получаю следующую позицию шара
            circle.x += round(circle.force[0] * 0.01)
            circle.y += round(circle.force[1] * 0.01)

            width, height = frame.shape[1::-1]
            # Если шар выходит за кадр, не дать ему уйти
            circle.x = max(0, min(width - 1, circle.x))
            circle.y = max(0, min(height - 1, circle.y))

            # Сделать так, чтобы шар отбивался от стенок
            if circle.x == 0 or circle.x == width - 1:
                circle.force[0] *= -1
            if circle.y == 0 or circle.y == height - 1:
                circle.force[1] *= -1


def optical_flow(frame, prev_frame, flow):
    return cv2.calcOpticalFlowFarneback(
        frame,
        prev_frame,
        flow,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


COLORS = [
    (244, 224, 222),
    (146, 235, 111),
    (119, 246, 193),
    (186, 235, 188),
    (143, 49, 116),
    (216, 156, 207),
    (231, 196, 167),
]


def random_circle(screen_width, screen_height):
    r = random.randint(int(screen_height / 40), int(screen_height / 16))
    x = random.randint(r, screen_width - r)
    y = random.randint(r, screen_height - r)
    # Выбираю случайный цвет из выбранного мной списка цветов
    color = COLORS[random.randint(0, len(COLORS) - 1)]
    return Circle(x, y, r, color)


def capture_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open the video cam")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"Frame size: {width} x {height}")

    window = "Circles"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    state = State(int(width), int(height))
    while True:
        success, frame = cap.read()
        if not success:
            print("Cannot read a frame from video stream")
            break
        cv2.imshow(window, state.process(frame))
        if cv2.waitKey(1) == 27:
            print("ESC key is pressed by user")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_from_camera()
