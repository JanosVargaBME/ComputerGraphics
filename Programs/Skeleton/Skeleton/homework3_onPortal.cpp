

//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Varga Janos
// Neptun : BJCZQ9
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
 
const int tessellationLevel = 20;
const int rotAngle = 1;
 
#define GLOBAL_OBJECT_COLOR vec3(0.78431, 0.78431, 0.78431)
 
float degreeToRad(float degree) {
	return (degree * M_PI) / 180;
}
 
template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};
 
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);}
 
typedef Dnum<vec2> Dnum2;
 
struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		setVariables();
	} 
 
	void setVariables() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}
 
	void Animate(float dt) {
		wEye = vec3((wEye.x - wLookat.x) * cos(dt) + (wEye.z - wLookat.z) * sin(dt) + wLookat.x,
			wEye.y,
			-(wEye.x - wLookat.x) * sin(dt) + (wEye.z - wLookat.z) * cos(dt) + wLookat.z);
	}
 
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}
 
	mat4 P() {
		return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
			        0,                      1 / tan(fov / 2), 0,                      0,
			        0,                      0,                -(fp + bp) / (bp - fp), -1,
			        0,                      0,                -2 * fp*bp / (bp - fp),  0);
	}
};
 
struct Material {
	vec3 kd, ks, ka;
	float shininess;
 
	Material(int a) {
		if (a > 0)
			kd = vec3(1.0f, 0.34902f, 0.0f);
		
		else 
			kd = vec3(0.48431, 0.48431, 0.48431);
		ka = kd * M_PI;
		ks = vec3(2, 2, 2);
		shininess = 50;
	}
};
 
struct Light {
	vec3 La, Le;
	vec4 wLightPos;
	
	void Animate() {
		mat4 rotationM = RotationMatrix(-degreeToRad(rotAngle), vec3(0, 1, 0));
		vec4 v = wLightPos;
		v = v * rotationM;
		wLightPos.x = v.x;
		wLightPos.y = v.y;
		wLightPos.z = v.z;
	}
};
 
struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	vec3	           wEye;
};
 
class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;
	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}
	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};
 
class PhongShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int   nLights;
		uniform vec3  wEye;
 
		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;
 
		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];
		out vec2 texcoord;
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
 
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";
 
	const char * fragmentSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
 
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};
 
		uniform Material material;
		uniform Light[8] lights;
		uniform int   nLights;
 
		uniform vec3 COLOR;
 
		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[8];
		in  vec2 texcoord;
		
        out vec4 fragmentColor;
 
		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			
			vec3 texColor = COLOR;
			
			//vec3 ka = material.ka * texColor;
			//vec3 kd = material.kd * texColor;
			vec3 ka = material.ka;
			vec3 kd = material.kd;
 
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += ka * lights[i].La + 
                           (kd * COLOR * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");
		setUniform(GLOBAL_OBJECT_COLOR, "COLOR");
		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};
 
class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
 
	virtual void AnimateShape(vec3& rotationAxis, vec3& translation) = 0;
};
 
class ParamSurface : public Geometry {
public:
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};
	unsigned int nVtxPerStrip, nStrips;
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;
	virtual VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}
	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
	virtual void AnimateShape(vec3& rotationAxis, vec3& translation) = 0;
};
 
enum ShapeType
{
	bottom,
	middle,
	top
};
 
class Sphere : public ParamSurface {
public:
	float r;
	ShapeType place;
	Sphere(float radius, ShapeType t) { r = radius; place = t; create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V) * r; Y = Sin(U) * Sin(V) * r; Z = Cos(V) * r;
	}
 
	void AnimateShape(vec3& rotationAxis, vec3& translation) {
		if (place == bottom)
			return;
		mat4 rotationM = RotationMatrix(-degreeToRad(rotAngle), vec3(0, 1, 0));
		vec4 v = vec4(translation.x, translation.y, translation.z, 1);
		v = v * rotationM;
		translation.x = v.x;
		translation.y = v.y;
		translation.z = v.z;
	}
};
 
class Plane : public ParamSurface {
public:
	float size;
	Plane(float s) { size = s; create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {}
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0, 1, 0);
		vd.position = vec3((u - 0.5) * 2, 0, (v - 0.5) * 2) * size;
		vd.texcoord = vec2(u, v);
		return vd;
	}
 
	void AnimateShape(vec3& rotationAxis, vec3& translation) {}
};
 
class Cylinder : public ParamSurface {
public:
	float r, height;
	ShapeType place;
	Cylinder(float radius, float _height, ShapeType t) { r = radius; height = _height; place = t; create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = V * height;
		X = Cos(U) * r; Y = V; Z = Sin(U) * r;
	}
 
	void AnimateShape(vec3& rotationAxis, vec3& translation) {
		if (place == bottom)
			return;
		mat4 rotationM = RotationMatrix(-degreeToRad(rotAngle), vec3(0, 1, 0));
		vec4 v = vec4(rotationAxis.x, rotationAxis.y, rotationAxis.z, 1);
		v = v * rotationM;
		rotationAxis.x = v.x;
		rotationAxis.y = v.y;
		rotationAxis.z = v.z;
 
		if (place == middle)
			return;
 
		rotationM = RotationMatrix(-degreeToRad(rotAngle), vec3(0, 1, 0));
		v = vec4(translation.x, translation.y, translation.z, 1);
		v = v * rotationM;
		translation.x = v.x;
		translation.y = v.y;
		translation.z = v.z;
	}
};
 
class Paraboloid : public ParamSurface {
public:
	float r, height;
	Paraboloid(float _r = 1.0f, float _h = 1.0f) { r = _r; height = _h; create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U  * M_PI; 
		V = V * 2.0f * M_PI;
		X = Pow(U, 0.5) * Cos(V) * r;
		Z = Pow(U, 0.5) * Sin(V) * r;
		Y = U * height;
	}
 
	void AnimateShape(vec3& rotationAxis, vec3& translation) {
		mat4 rotationM = RotationMatrix(-degreeToRad(rotAngle), vec3(0, 1, 0));
		vec4 v = vec4(translation.x, translation.y, translation.z, 1);
		v = v * rotationM;
		translation.x = v.x;
		translation.y = v.y;
		translation.z = v.z;
	}
};
 
struct Object {
	Shader *   shader;
	Material * material;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		mat4 RotMat;
		RotMat = RotationMatrix(-rotationAngle, rotationAxis);
		M = ScaleMatrix(scale) * RotMat * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotMat * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}
 
	virtual void Animate() {
		geometry->AnimateShape(rotationAxis, translation);
	}
};
 
class Scene {
	std::vector<Object *> objects;
	std::vector<Light> lights;
public:
	Camera camera;
	void setUp() {
		Shader* phongShader = new PhongShader();
		Material* material0 = new Material(1);
		Geometry* sphere1 = new Sphere(0.2f, bottom);
		Geometry* sphere2 = new Sphere(0.2f, middle);
		Geometry* sphere3 = new Sphere(0.2f, top);
		Geometry* cylinder1 = new Cylinder(1.0f, 0.5f, bottom);
		Geometry* cylinder2 = new Cylinder(0.1f, 3.5f, middle);
		Geometry* cylinder3 = new Cylinder(0.1f, 1.0f, top);
		Geometry* paraboloid = new Paraboloid(0.5f, 0.5f);
		Geometry* plane = new Plane(100);
 
		float offSetOnY = -0.43f;
		
		Object* cylinderObject1 = new Object(phongShader, material0, cylinder1);
		cylinderObject1->rotationAngle = degreeToRad(-10);
		cylinderObject1->translation = vec3(0, -0.4 + offSetOnY, 0);
		cylinderObject1->rotationAxis = vec3(1, 0, 0);
		
 
		Object* sphereObject1 = new Object(phongShader, material0, sphere1);
		sphereObject1->translation = vec3(0, 0.15 + offSetOnY, 0);
		
		Object* cylinderObject2 = new Object(phongShader, material0, cylinder2);
		cylinderObject2->rotationAngle = degreeToRad(30);
		cylinderObject2->rotationAxis = vec3(0, 0, 1);
		cylinderObject2->translation = vec3(0, 0.3 + offSetOnY, 0);
		
		Object* sphereObject2 = new Object(phongShader, material0,  sphere2);
		sphereObject2->translation = vec3(1.8, 3.3 + offSetOnY, 0);
 
		Object* cylinderObject3 = new Object(phongShader, material0,  cylinder3);
		cylinderObject3->rotationAxis = vec3(0, 0, 1);
		cylinderObject3->rotationAngle = degreeToRad(100);
		cylinderObject3->translation = vec3(1.8, 3.3 + offSetOnY, 0);
 
		Object* sphereObject3 = new Object(phongShader, material0,  sphere3);
		sphereObject3->translation = vec3(2.8, 3.1 + offSetOnY, 0);
 
		Material* mat2 = new Material(-1);
		Object* planeObject = new Object(phongShader, mat2,  plane);
		planeObject->translation = vec3(0, -1, 0);
		
		Object* paraboloidObject = new Object(phongShader, material0,  paraboloid);
		paraboloidObject->translation = vec3(2.8, 3 + offSetOnY, 0);
		paraboloidObject->rotationAxis = vec3(1, 0, 0);
		paraboloidObject->rotationAngle = degreeToRad(180);
 
		lights.resize(1);
		lights[0].wLightPos = vec4(2.8, 2.9 + offSetOnY, 0, 1);
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(1, 1, 1);
 
		objects.push_back(planeObject);
		objects.push_back(cylinderObject1);
		objects.push_back(sphereObject1);
		objects.push_back(cylinderObject2);
		objects.push_back(sphereObject2);
		objects.push_back(cylinderObject3);
		objects.push_back(sphereObject3);
		objects.push_back(paraboloidObject);
	}
 
	void Build() {
		setUp();
		camera.wEye = vec3(0, 0, 8);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);
	}
 
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}
 
	void Animate() {
		lights[0].Animate();
		for (Object * obj : objects) obj->Animate();
	}
};
 
Scene scene;
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}
void onDisplay() {
	glClearColor(0.82353f, 0.82353f, 0.82353f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}
void onKeyboard(unsigned char key, int pX, int pY) { }
void onKeyboardUp(unsigned char key, int pX, int pY) { }
void onMouse(int button, int state, int pX, int pY) { }
void onMouseMotion(int pX, int pY) {}
 
int timesRan = 0;
 
void onIdle() {
	float tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
 
	timesRan++;
	if (timesRan == 0) {
		for (int i = 0; i < 360; i++)
			scene.Animate();
	}
 
	if (timesRan > 2) {
		for (int i = 0; i < 80; i++)
			scene.Animate();
	}
 
	for (int i = 0; i < 10; i++)
		scene.camera.Animate(tend / 500.0f);
	
	glutPostRedisplay();

}

