#include "framework.h"
 
const char* const vertexSource = R"(
	#version 330
	precision highp float;
 
	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;
 
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
	}
)";
 
const char* const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	out vec4 outColor;
 
	void main() {
		outColor = vec4(color, 1);
	}
)";
 
GPUProgram gpuProgram;
unsigned int vao;
 
#define POINT_SIZE 10
#define LINE_SIZE 3
 
#define TOLERANCE 0.02f
 
#define yellow vec3(1.0f, 1.0f, 0.0f)
#define red vec3(1.0f, 0.0f, 0.0f)
#define cian vec3(0.0f, 1.0f, 1.0f)
#define white vec3(1.0f, 1.0f, 1.0f)
 
unsigned char actualPressedKey = ' ';
 
struct Point {
	vec2 coord;
	Point() {}
	Point(vec2 _coord) { coord = _coord;}
	float distanceFromPoint(Point pont) {
		return sqrtf((pont.coord.x - coord.x) * (pont.coord.x - coord.x) + (pont.coord.y - coord.y) * (pont.coord.y - coord.y));
	}
};
 
std::vector<Point> points;
 
struct Line {
	vec2 startCoord;
	vec2 endCoord;
	vec2 normal;
	float equationResult;
	Line() {
		startCoord = vec2(-10, -10);
		endCoord = vec2(-10, -10);
		equationResult = 0;
	}
	Line(vec2 _startCoord, vec2 _endCoord) {
		startCoord = _startCoord;
		endCoord = _endCoord;
		vec2 direction = _endCoord - _startCoord;
		normal = vec2(direction.y, -direction.x);
		equationResult = normal.x * startCoord.x + normal.y * startCoord.y;
	}
	Line(Point p, vec2 _normal) {
		startCoord = p.coord;
		normal = _normal;
		equationResult = normal.x * startCoord.x + normal.y * startCoord.y;
	}
 
	void makeLineToLongLine() {
		int pointsUpdated = 0;
		Line edge = Line(vec2(0.0f, 1.0f), vec2(1.0f, 1.0f));
		vec2 intersect = getIntersection(edge);
		if (intersect.x >= -1.0 && intersect.x <= 1.0) {
			startCoord = intersect;
			pointsUpdated++;
		}
 
		edge = Line(vec2(0.0f, -1.0f), vec2(1.0f, -1.0f));
		intersect = getIntersection(edge);
		if (intersect.x >= -1.0 && intersect.x <= 1.0) {
			if (pointsUpdated == 0) {
				startCoord = intersect;
			}
			else {
				endCoord = intersect;
				return;
			}
			pointsUpdated++;
		}
 
		edge = Line(vec2(-1.0f, 0.0f), vec2(-1.0f, 1.0f));
		intersect = getIntersection(edge);
		if (intersect.y >= -1.0 && intersect.y <= 1.0) {
			if (pointsUpdated == 0) {
				startCoord = intersect;
			}
			else {
				endCoord = intersect;
				return;
			}
			pointsUpdated++;
		}
 
		edge = Line(vec2(1.0f, 0.0f), vec2(1.0f, 1.0f));
		intersect = getIntersection(edge);
		if (intersect.y >= -1.0 && intersect.y <= 1.0) {
			if (pointsUpdated == 0) {
				startCoord = intersect;
			}
			else {
				endCoord = intersect;
				return;
			}
			pointsUpdated++;
		}
	}
 
	vec2 getIntersection(Line otherLine) {
		float det = normal.x * otherLine.normal.y - normal.y * otherLine.normal.x;
 
		if (det != 0) {
			float x = (equationResult * otherLine.normal.y - normal.y * otherLine.equationResult)/det;
			float y = (normal.x * otherLine.equationResult - equationResult * otherLine.normal.x)/det;
			if((x<=1 && x>=-1) && (y<=1 && y>=-1))
				return vec2(x, y);
		}
		return vec2(-10,-10);
	}
};
 
std::vector<Line> lines;
 
struct Circle {
	vec2 center;
	float R;
	Circle() { R = 0.0f; }
	Circle(vec2 _center, float _R) {
		center = _center;
		R = _R;
	}
	
	std::vector<Point> getIntersections(Line line) {
		Point c = Point(center);
		Line test = Line(c, vec2(line.normal.y, -line.normal.x));
 
		Point intersect = test.getIntersection(line);
 
		float distance = intersect.distanceFromPoint(c);
		std::vector<Point> result;
		if (distance > R + TOLERANCE) {
			return result;
		}
		vec2 start = vec2(line.startCoord.x - center.x, line.startCoord.y - center.y);
		vec2 end = vec2(line.endCoord.x - center.x, line.endCoord.y - center.y);
		Line lineCopy = Line(start, end);
 
		float A, B, C;
		A = lineCopy.normal.x;
		B = lineCopy.normal.y;
		C = -lineCopy.equationResult;
 
		float x0 = -A * C / (A * A + B * B);
		float y0 = -B * C / (A * A + B * B);
 
		if (distance >= R - TOLERANCE && distance <= R + TOLERANCE) {
			x0 += center.x;
			y0 += center.y;
			result.push_back(Point(vec2(x0, y0)));
			return result;
		}
 
		float d = R * R - C * C / (A * A + B * B);
		float mult = sqrtf(d / (A * A + B * B));
		float ax, ay, bx, by;
 
		ax = x0 + B * mult;
		bx = x0 - B * mult;
		ay = y0 - A * mult;
		by = y0 + A * mult;
 
		ax += center.x;
		ay += center.y;
		bx += center.x;
		by += center.y;
		result.push_back(Point(vec2(ax, ay)));
		result.push_back(Point(vec2(bx, by)));
		//source: https://cp-algorithms.com/geometry/circle-line-intersection.html
		return result;
	}
 
	std::vector<Point> getIntersections(Circle otherCircle) {
		std::vector<Point> result;
		Point first = Point(center);
		Point last = Point(otherCircle.center);
		float Dist = first.distanceFromPoint(last);
		if (Dist > R + otherCircle.R || Dist == 0.0f)
			return result;
 
		float u = center.x;
		float v = center.y;
		float r = R;
 
		float p = otherCircle.center.x;
		float q = otherCircle.center.y;
		float k = otherCircle.R;
 
		float A = (u * u + v * v - r * r) - (p * p + q * q - k * k);
		float E = A / 2;
		float C = u - p;
		if (C == 0.0f)
			return result;
 
		float D = v - q;
 
		float W = E / C;
		float Z = D / C;
 
		float F = W - p;
		float G = q * q - k * k;
 
		float a = Z * Z + 1;
		float b = -(2 * F * Z + 2 * q);
		float c = F * F + G;
 
		float det = (b * b - 4 * a * c);
		float x1, y1, x2, y2;
		if (det == 0.0f) {
			if (a != 0.0f) {
				y1 = (-b) / (2 * a);
				x1 = W - Z * y1;
				result.push_back(Point(vec2(x1, y1)));
			}
		}
		else {
			if (a != 0.0f) {
				y1 = (-b + sqrtf(det)) / (2 * a);
				y2 = (-b - sqrtf(det)) / (2 * a);
 
				x1 = W - Z * y1;
				x2 = W - Z * y2;
 
				result.push_back(Point(vec2(x1, y1)));
				result.push_back(Point(vec2(x2, y2)));
			}
		}
 
		return result;
	}
};
 
std::vector<Circle> circles;
 
struct Click {
	float x;
	float y;
	float rad = TOLERANCE;
	unsigned char pressedKey = ' ';
	Point point;
	Line line;
	Circle circle;
	Click() {
		x = 0.0f;
		y = 0.0f;
		point = Point(vec2(-10, -10));
		line = Line(vec2(-10, -10), vec2(-10, -10));
		circle = Circle(vec2(-10, -10), -10);
	}
 
	Click(float _x, float _y) {
		x = _x;
		y = _y;
		pressedKey = actualPressedKey;
		point = Point(vec2(-10, -10));
		line = Line(vec2(-10, -10), vec2(-10, -10));
		circle = Circle(vec2(-10, -10), 10);
	}
 
	bool closeToPoint(Point pont) {
		if (pont.coord.x <= (x + rad) && pont.coord.x >= (x - rad)) {
			if (pont.coord.y <= (y + rad) && pont.coord.y >= (y - rad))
				return true;
		}
 
		return false;
	}
 
	bool closeToLine(Line f) {
		Point cl = Point(vec2(x, y));
		Line e = Line(cl, vec2(f.normal.y, -f.normal.x));
		Point intersect = Point(e.getIntersection(f));
		float distance = intersect.distanceFromPoint(cl);
		if (distance < rad)
			return true;
		return false;
	}
 
	bool closeToCircle(Circle c) {
		Point center = Point(c.center);
		float distance = center.distanceFromPoint(Point(vec2(x, y)));
		if (distance <= (c.R + rad) && distance >= (c.R - rad)) {
			return true;
		}
		return false;
	}
 
	float distanceFromClick(Click otherClick) {
		return sqrtf((otherClick.x - x) * (otherClick.x - x) + (otherClick.y - y) * (otherClick.y - y));
	}
};
 
std::vector<Click> clicks;
 
struct WhiteObject {
	char type;
	int index;
	WhiteObject(char _type, int _index) { type = _type; index = _index; }
};
 
std::vector<WhiteObject> toWhite;
 
void drawObject(int type, std::vector<vec2> vertices, vec3 color) {
 
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, color.x, color.y, color.z);
 
	glBufferData(GL_ARRAY_BUFFER,
		vertices.size() * sizeof(vec2),
		&vertices[0],
		GL_STATIC_DRAW);
 
	glDrawArrays(type, 0, vertices.size());
}
 
std::vector<vec2> linesToVec2Vector(std::vector<Line> linesList) {
	std::vector<vec2> result;
	for (int i = 0; i < linesList.size(); i++) {
		result.push_back(linesList.at(i).startCoord);
		result.push_back(linesList.at(i).endCoord);
	}
	return result;
}
 
std::vector<vec2> pointsToVec2Vector(std::vector<Point> pointsList) {
	std::vector<vec2> result;
	for (int i = 0; i < pointsList.size(); i++)
		result.push_back(pointsList.at(i).coord);
	return result;
}
 
std::vector<vec2> circlesToVec2Vector(std::vector<Circle> circlesList) {
	std::vector<vec2> result;
	float x, y;
	for (int i = 0; i < circlesList.size(); i++) {
		for (float angle = 0; angle < 360; angle += 1.0f) {
			x = circlesList.at(i).center.x + circlesList.at(i).R * cosf(angle);
			y = circlesList.at(i).center.y + circlesList.at(i).R * sinf(angle);
			result.push_back(vec2(x, y));
		}
	}
	return result;
}
 
void removeNotNeededThings() {
	for (int i = 0; i < lines.size(); i++) {
		bool remove = false;
		if (lines.at(i).startCoord.x < -1 || lines.at(i).startCoord.x > 1)
			remove = true;
		else if(lines.at(i).startCoord.y < -1 || lines.at(i).startCoord.y > 1)
			remove = true;
		else if(lines.at(i).endCoord.x < -1 || lines.at(i).endCoord.x > 1)
			remove = true;
		else if(lines.at(i).endCoord.y < -1 || lines.at(i).endCoord.y > 1)
			remove = true;
 
		if (remove) {
			lines.erase(lines.begin() + i);
			i--;
		}
	}
 
	for (int i = 0; i < circles.size(); i++) {
		bool remove = false;
		if (circles.at(i).center.x < -1 || circles.at(i).center.x > 1)
			remove = true;
		else if (circles.at(i).center.y < -1 || circles.at(i).center.y > 1)
			remove = true;
		else if (circles.at(i).R <= 0.0f)
			remove = true;
 
		if (remove) {
			circles.erase(circles.begin() + i);
			i--;
		}
	}
 
	for (int i = 0; i < points.size(); i++) {
		bool remove = false;
		if (points.at(i).coord.x < -1 || points.at(i).coord.x > 1)
			remove = true;
		else if(points.at(i).coord.y < -1 || points.at(i).coord.y > 1)
			remove = true;
 
		if (remove) {
			points.erase(points.begin() + i);
			i--;
		}
	}
}
 
void checkOperations() {
	if (clicks.size() < 2)
		return;
	if (clicks.size() >= 2) {
		Click first = clicks.at(clicks.size() - 1);
		Click last = clicks.at(clicks.size() - 2);
		if (first.pressedKey == 'l') {
			if (last.pressedKey == 'l') {
				Line created = Line(vec2(first.point.coord.x, first.point.coord.y), vec2(last.point.coord.x, last.point.coord.y));
				created.makeLineToLongLine();
				lines.push_back(created);
				clicks.clear();
				toWhite.clear();
				return;
			}
		}
 
		else if (first.pressedKey == 'i') {
			if (last.pressedKey == 'i') {
				if (first.line.startCoord.x >= -1 && last.line.startCoord.x >= -1) {
					Point created = Point(first.line.getIntersection(last.line));
					if (created.coord.x >= -1)
						points.push_back(created);
 
				}
				else {
					std::vector<Point> result;
					if (first.line.startCoord.x >= -1 && last.circle.center.x >= -1)
						result = last.circle.getIntersections(first.line);
						
					else if (first.circle.center.x >= -1 && last.line.startCoord.x >= -1)
						result = first.circle.getIntersections(last.line);
						
					else if (first.circle.center.x >= -1 && last.circle.center.x >= -1) {
						if (first.circle.center.x == 0.0f && first.circle.center.y == 0.0f)
							result = first.circle.getIntersections(last.circle);
						else if (last.circle.center.x == 0.0f && last.circle.center.y == 0.0f)
							result = last.circle.getIntersections(first.circle);
						else
							result = first.circle.getIntersections(last.circle);
					}
					
					points.insert(std::end(points), std::begin(result), std::end(result));
 
				}
				toWhite.clear();
				clicks.clear();
				return;
			}
		}
	}
	if (clicks.size() > 2) {
		if (clicks.at(clicks.size() - 1).pressedKey == 'c') {
			if (clicks.at(clicks.size() - 3).pressedKey == 's' && clicks.at(clicks.size() - 2).pressedKey == 's') {
				Click firstClick = clicks.at(clicks.size() - 3);
				Click secondClick = clicks.at(clicks.size() - 2);
				Click center = clicks.at(clicks.size() - 1);
				float radius = secondClick.distanceFromClick(firstClick);
				circles.push_back(Circle(vec2(center.point.coord.x, center.point.coord.y), radius));
				clicks.pop_back();
				toWhite.pop_back();
				return;
			}
		}
	}
}
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
 
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
 
	unsigned int vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
 
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
 
	lines.push_back(Line(vec2(-1.0f, 0.0f), vec2(1.0f, 0.0f)));
 
	points.push_back(Point(vec2(0.0f, 0.0f)));
	points.push_back(Point(vec2(0.2f, 0.0f)));
 
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}
 
void addClickIfNearToPoint(Click actualClick) {
 
	if (clicks.size() > 0 && toWhite.size() > 0) {
		char lastChar = clicks.at(clicks.size() - 1).pressedKey;
		if (lastChar == 's' && (actualClick.pressedKey != 'c' && actualClick.pressedKey != 's')) {
			toWhite.clear();
		}
		else if (lastChar != actualClick.pressedKey) {
			toWhite.clear();
		}
	}
 
	if (actualClick.pressedKey == 's' || actualClick.pressedKey == 'c' || actualClick.pressedKey == 'l') {
		for (int i = 0; i < points.size(); i++) {
			if (actualClick.closeToPoint(points.at(i))) {
				actualClick.point = points.at(i);
				clicks.push_back(actualClick);
				toWhite.push_back(WhiteObject('p', i));
				return;
			}
		}
	}
	if (actualClick.pressedKey == 'i') {
		for (int i = 0; i < lines.size(); i++) {
			if (actualClick.closeToLine(lines.at(i))) {
				actualClick.line = lines.at(i);
				clicks.push_back(actualClick); 
				toWhite.push_back(WhiteObject('l', i));
				return;
			}
		}
		for (int i = 0; i < circles.size(); i++) {
			if (actualClick.closeToCircle(circles.at(i))) {
				actualClick.circle = circles.at(i);
				clicks.push_back(actualClick);
				toWhite.push_back(WhiteObject('c', i));
				return;
			}
		}
	}
}
 
void drawWhiteThings() {
	std::vector<Circle> cToWhite;
	std::vector<Line> lToWhite;
	std::vector<Point> pToWhite;
	for (int i = 0; i < toWhite.size(); i++) {
		if (toWhite.at(i).type == 'p')
			pToWhite.push_back(points.at(toWhite.at(i).index));
 
		else if (toWhite.at(i).type == 'l')
			lToWhite.push_back(lines.at(toWhite.at(i).index));
 
		else if (toWhite.at(i).type == 'c')
			cToWhite.push_back(circles.at(toWhite.at(i).index));
	}
 
	glPointSize(LINE_SIZE);
	drawObject(GL_POINTS, circlesToVec2Vector(cToWhite), white);
	glLineWidth(LINE_SIZE);
	drawObject(GL_LINES, linesToVec2Vector(lToWhite), white);
	glPointSize(POINT_SIZE);
	drawObject(GL_POINTS, pointsToVec2Vector(pToWhite), white);
 
}
 
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
 
	float MVPtransf[4][4] = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
 
	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
 
	glBindVertexArray(vao);
 
	glPointSize(LINE_SIZE);
	drawObject(GL_POINTS, circlesToVec2Vector(circles), cian);
	glLineWidth(LINE_SIZE);
	drawObject(GL_LINES, linesToVec2Vector(lines), red);
	glPointSize(POINT_SIZE);
	drawObject(GL_POINTS, pointsToVec2Vector(points), yellow);
 
	drawWhiteThings();
 
	glutSwapBuffers();
}
 
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd')
		glutPostRedisplay();
	actualPressedKey = key;
}
 
void onKeyboardUp(unsigned char key, int pX, int pY) {}
 
void onMouseMotion(int pX, int pY) {}
 
void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}
	Click actualClick = Click(cX, cY);
	char characters[] = { 's', 'c', 'l', 'i'};
	if (buttonStat == "released") {
		for (int i = 0; i < sizeof(characters); i++) {
			if (actualPressedKey == characters[i])
				addClickIfNearToPoint(actualClick);
		}
	}
	checkOperations();
	removeNotNeededThings();
	glutPostRedisplay();
}
 
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);

}

