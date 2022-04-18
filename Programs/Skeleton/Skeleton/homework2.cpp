#include "framework.h"
 
const float epsilon = 0.0001f;
const float radiusGlobal = 0.03f;
int prev = 1;
#define ORIGO vec3(0.0f, 0.0f, 0.0f);
 
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};
 
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};
 
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
 
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
	virtual void translate(const vec3& coord) = 0;
	vec3 allRotateCalcCenter(float _angle, vec3& point ,vec3& axis) {
		float x = point.x;
		float y = point.y;
		float z = point.z;
 
		float angle = _angle * (M_PI / 180);
 
		mat4 rotMat = RotationMatrix(angle, axis);
 
		vec4 result = vec4(x, y, z, 1) * rotMat;
 
		x = result.x;
		y = result.y;
		z = result.z;
 
		return vec3(x, y, z);
	}
	virtual void rotate(float angle, vec3 axis) = 0;
	virtual void setCut(float x1, float x2){}
	virtual vec3 getCenter() { return vec3(0, 0, 0); }
};
 
struct Sphere : public Intersectable {
	vec3 center;
	float radius;
 
	Sphere(Material* _material) {
		center = ORIGO;
		radius = radiusGlobal;
		material = _material;
	}
 
	virtual void translate(const vec3& coord) {center = coord;}
 
	vec3 getCenter() { return center; }
 
	void rotate(float angle, vec3 axis) {center = allRotateCalcCenter(angle, center, axis);}
 
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};
 
struct Quadrics : public Intersectable {
	mat4 Q;
	float zmin, zmax;
	vec3 translation;
 
	Quadrics(Material* _material, mat4& _Q, float _zmin, float _zmax) {
		Q = _Q;
		zmin = _zmin;
		zmax = _zmax;
		material = _material;
		translation = ORIGO;
	}
 
	void setCut(float x1, float x2) {
		zmin = x1;
		zmax = x2;
	}
 
	virtual void translate(const vec3& coord) {translation = coord;}
 
	void rotate(float _angle, vec3 axis) {
		translation = allRotateCalcCenter(_angle, translation, axis);
	}
	vec3 gradf(vec3 r) {
		vec4 rh(r.x, r.y, r.z, 1);
		vec4 g = rh * Q * 2.0f;
		return vec3(g.x, g.y, g.z);
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = ray.start - translation;
		vec4 S(start.x, start.y, start.z, 1.0f), D(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
		float a = dot(D * Q, D);
		float b = dot(S * Q, D) + dot(D * Q, S);
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
 
		if (discr < 0)
			return hit;
 
		float sqrt_discr = sqrtf(discr);
 
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
 
		
		float cutEdges = 0.0f;
 
		if (Q.rows[0][0] == 0)
			cutEdges = p1.x;
		else if(Q.rows[1][1] == 0)
			cutEdges = p1.y;
		else if (Q.rows[2][2] == 0)
			cutEdges = p1.z;
 
		if (cutEdges < zmin || cutEdges > zmax)
			t1 = -1.0f;
		
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
 
		
		if (Q.rows[0][0] == 0)
			cutEdges = p2.x;
		else if (Q.rows[1][1] == 0)
			cutEdges = p2.y;
		else if (Q.rows[2][2] == 0)
			cutEdges = p2.z;
 
		if (cutEdges < zmin || cutEdges > zmax)
			t2 = -1.0f;
		
		if (t1 <= 0.0f && t2 <= 0.0f)
			return hit;
		if (t1 <= 0.0f)
			hit.t = t2;
		else if (t2 <= 0.0f)
			hit.t = t1;
		else if (t2 < t1)
			hit.t = t2;
		else
			hit.t = t1;
 
		hit.position = start + ray.dir * hit.t;
		hit.normal = normalize(gradf(hit.position));
		hit.position = hit.position + translation;
		hit.material = material;
		
		return hit;
	}
};
 
struct Plane : public Intersectable {
	vec3 point, normal;
 
	Plane(Material* mat, const vec3& _normal) {
		point = ORIGO;
		material = mat;
		normal = normalize(_normal);
	}
 
	void rotate(float angle, vec3 axis) {
		point = allRotateCalcCenter(angle, point, axis);
	}
 
	virtual void translate(const vec3& coord) {
		point = coord;
	}
 
	Hit intersect(const Ray& ray) {
		Hit hit;
		double NdotV = dot(normal, ray.dir);
		if (fabs(NdotV) < epsilon) return hit;
		double t = dot(normal, point - ray.start) / NdotV;
		if (t < epsilon) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		hit.material = material;
		return hit;
	}
};
 
class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};
 
struct Light {
	vec3 position;
	vec3 Le;
	Light(vec3 _pos, vec3 _Le = vec3(2,2,2)) {
		Le = _Le;
		position = _pos;
	}
	void rotate(float _angle, const vec3& axis) {
		float x = position.x;
		float y = position.y;
		float z = position.z;
 
		float angle = _angle * (M_PI / 180);	//TO RADIAN
 
		mat4 rotMat = RotationMatrix(angle, axis);
 
		vec4 result = vec4(x, y, z, 1) * rotMat;
 
		position.x = result.x;
		position.y = result.y;
		position.z = result.z;
	}
};
 
class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	vec3 La;
public:
	Camera camera;
 
	void setUpObjects() {
		mat4 cylinderBig = ScaleMatrix(vec3(-40, 0, -40));
		mat4 cylinderSmallH = ScaleMatrix(vec3(-3500, 0, -3500));
		mat4 cylinderSmallV = ScaleMatrix(vec3(0, -3500, -3500));
 
		float quad = 6.0f;
		float width = 0.4f;
 
		mat4 paraboloidM = mat4(
			quad,	0,		0,		0,
			0,		0,		0,		width,
			0,		0,	quad,		0,
			0,	width,		0,		0
		);
 
		Material*	mt	= new Material(vec3(0.15196f, 0.15196f, 0.15196f), vec3(2, 2, 2), 50);
		Material*	mt2 = new Material(vec3(0.4f, 0.3f, 0.2f), vec3(2, 2, 2), 50);
		Material*	mt3 = new Material(vec3(0.66275f, 0.66275f, 0.66275f), vec3(2, 2, 2), 50);
 
		Plane*		pl1	= new Plane(mt2, vec3(0, 1, 0));
		Plane*		pl2 = new Plane(mt3, vec3(0, 0, 1));
		Quadrics*	cy1 = new Quadrics(mt,	cylinderBig,	-0.2, -0.15);
		Quadrics*	cy2 = new Quadrics(mt,	cylinderSmallH,	-0.08, 0.4);
		Quadrics*	cy3 = new Quadrics(mt,	cylinderSmallV,	-0.4, -0.02);
		Quadrics*	pb1 = new Quadrics(mt,	paraboloidM,	0.1, 0.4);
		Sphere*		sp1	= new Sphere(mt);
		Sphere*		sp2	= new Sphere(mt);
		Sphere*		sp3	= new Sphere(mt);
 
		objects.push_back(pl1);
		objects.push_back(cy1);
		objects.push_back(sp1);
		objects.push_back(cy2);
		objects.push_back(sp2);
		objects.push_back(cy3);
		objects.push_back(sp3);
		objects.push_back(pb1);
 
		objects[0]->translate(vec3(0, -0.2f, 0));
		objects[1]->translate(vec3(0.0, 0.0, 0));
		objects[2]->translate(vec3(0.0, -0.11, 0));
		objects[3]->translate(vec3(0.0, 0.0, 0));
		objects[4]->translate(vec3(0.0, 0.43, 0));
		objects[5]->translate(vec3(0.0, 0.43, 0));
		objects[6]->translate(vec3(-0.43, 0.43, 0));
		objects[7]->translate(vec3(-0.43, 0.4, 0));
 
	}
 
	void afterTime() {
		objects[7]->rotate(90.0f, vec3(0, 1, 0));
		objects[6]->rotate(90.0f, vec3(0, 1, 0));
		lights[0]->rotate(90.0f, vec3(0, 1, 0));
 
		float x = objects[6]->getCenter().x;
		float y = objects[6]->getCenter().y;
		float z = objects[6]->getCenter().z;
 
		Material* mt = new Material(vec3(0.15196f, 0.15196f, 0.15196f), vec3(2, 2, 2), 50);
		mat4 cylinderSmallV = ScaleMatrix(vec3(0, -3500, -3500));
		Quadrics* cy3 = new Quadrics(mt, cylinderSmallV, -0.4, -0.02);
 
		if (x < -0.01f) {
			cy3->Q = ScaleMatrix(vec3(0, -3500, -3500));
			cy3->zmin = -0.4;
			cy3->zmax = -0.02;
			cy3->translate(vec3(0.0, 0.43, 0));
		}
		else if (x > 0.01f) {
			cy3->Q = ScaleMatrix(vec3(0, -3500, -3500));
			cy3->zmin = 0.02;
			cy3->zmax = 0.4;
			cy3->translate(vec3(0.0, 0.43, 0));
		}
		else if (z < -0.01f) {
			cy3->Q = ScaleMatrix(vec3(-3500, -3500, 0));
			cy3->zmin = -0.4;
			cy3->zmax = -0.02;
			cy3->translate(vec3(0.0, 0.43, 0));
		}
		else if (z > 0.01f) {
			cy3->Q = ScaleMatrix(vec3(-3500, -3500, 0));
			cy3->zmin = 0.02;
			cy3->zmax = 0.4;
			cy3->translate(vec3(0.0, 0.43, 0));
		}
		objects[5] = cy3;
	}
 
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		La = vec3(0.4f, 0.4f, 0.4f);
		lights.push_back(new Light(vec3(-0.43f, 0.20f, 0.0f)));
		setUpObjects();
	}
 
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
 
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
 
	bool shadowIntersectWithPointLight(Ray ray, Light* light) {
		for (Intersectable* obj : objects) {
			float interPointBefoAft = obj->intersect(ray).t;
			vec3 v = light->position - ray.start;
			float distance = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
			if (interPointBefoAft < distance && interPointBefoAft > 0)	return true;
		}
		return false;
	}
 
	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			vec3 lDirectionVector = light->position - hit.position;
			float distFromlight = sqrtf(lDirectionVector.x * lDirectionVector.x + lDirectionVector.y * lDirectionVector.y + lDirectionVector.z * lDirectionVector.z);
			lDirectionVector = normalize(lDirectionVector);
 
			vec3 lightIntensity = light->Le * (1.0f / (1.0f + distFromlight * distFromlight));
 
			Ray shadowRay(hit.position + hit.normal * epsilon * 1.0f, lDirectionVector);
 
			float cosTheta = dot(hit.normal, lDirectionVector);
 
			if (cosTheta > 0 && !shadowIntersectWithPointLight(shadowRay, light)) {
 
				outRadiance = outRadiance + lightIntensity * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + lDirectionVector);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0)
					outRadiance = outRadiance + lightIntensity * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};
 
GPUProgram gpuProgram;
Scene scene;
 
const char* vertexSource = R"(
	#version 330
    precision highp float;
 
	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;
 
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
}
)";
 
const char* fragmentSource = R"(
	#version 330
    precision highp float;
 
	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
 
	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";
 
class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
 
		unsigned int vbo;
		glGenBuffers(1, &vbo);
 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}
 
	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};
 
FullScreenTexturedQuad* fullScreenTexturedQuad;
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
 
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
 
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
 
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
 
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}
 
void onKeyboard(unsigned char key, int pX, int pY) {}
 
void onKeyboardUp(unsigned char key, int pX, int pY) {}
 
void onMouse(int button, int state, int pX, int pY) {}
 
void onMouseMotion(int pX, int pY) {}
 
void onIdle() {
 
	float t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	int x = (int)t;
	if (x != prev) {
		scene.afterTime();
 
		std::vector<vec4> image(windowWidth * windowHeight);
		scene.render(image);
		fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
		prev=x;
	}
	scene.camera.Animate(t / 1000.0f);
	glutPostRedisplay();

}

