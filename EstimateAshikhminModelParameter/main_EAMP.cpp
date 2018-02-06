#include "BRDFAxisTable.h"
#include "OGMArms.h"
#include "bmp.h"
#include "exr.h"
#include "hdr.h"
#include "MathUtil.h"
#include "vector.h"
#include "matrix.h"

// range warp and weft
size_t xs_warp	=	390;
size_t xe_warp	=	410;
size_t ys_warp	=	295;
size_t ye_warp	=	315;
size_t xs_weft	=	340;
size_t xe_weft	=	360;
size_t ys_weft	=	300;
size_t ye_weft	=	315;

float n = 1.5f;

// path
std::string inputpath		=	"E:\\My Research\\cadcg\\OGM3\\20090819";
std::string materialprefix	=	"nekutai_";
std::string whiteprefix		=	"white_";

//
const size_t maxnumT	=	19;
const size_t maxT		=	80;
const size_t maxP		=	355;
const size_t T	=	(maxT/5+1);
const size_t P	=	(maxP/5+1);
size_t image_number	=	T*P;

array2d<int> noshadow(T,P);

//
// mist
//
#include "mist/numeric.h"

typedef mist::matrix<float> MAT;
typedef MAT UP;
float norm(UP u){
	float d=0.0f;
	for(size_t s=0; s<u.size(); s++)
		d += u[s]*u[s];
	return sqrt(d);
}
const size_t Nu = 5;
float du[5] = { 0.04f, 0.04f, 0.01f, 0.1f, 0.2f };

#include <iostream>
#include <fstream>
#include "pca.h"

using namespace tkd;

const int max_t=80;
const int max_p=360;
array2d<float> refR(85/5,360/5);
array2d<float> refG(85/5,360/5);
array2d<float> refB(85/5,360/5);
array2d<float> srefR(85/5,360/5);
array2d<float> srefG(85/5,360/5);
array2d<float> srefB(85/5,360/5);
array2d<float> srefY(85/5,360/5);
Vec3 N(0.0f,0.0f);

const float min_sigma = 1e-10f;
const float dark = 1e-10f;
float threshold = 0.0f;

struct AMP {
	AMP() : n(), t(), rho_s(0.0f), sigma_x(min_sigma), sigma_y(min_sigma) {};
	Vec3fs n;
	Vec3 t;
//	RGB<float> rho_s;
//	RGB<float> sigma_x;
//	RGB<float> sigma_y;
	float rho_s;
	float sigma_x;
	float sigma_y;
};

float Phi(float p);
inline std::ostream &operator <<( std::ostream &out, const AMP &amp )
{
	out << "n(" << degrees(amp.n.theta) << ", " << Phi(degrees(amp.n.phi)) << "), ";
	out << "t(" << degrees(amp.t.GetTheta()) << ", " << Phi(degrees(amp.t.GetPhi())) << "), ";
	out << "rho_s : " << amp.rho_s << ", ";
	out << "sigma_x : " << amp.sigma_x << ", ";
	out << "sigma_y : " << amp.sigma_y;
	return( out );
}
float AshikhminModel(UP u, Vec3 H){

	float v1 = pow( cos( H.GetPhi( ) ) / u[3], 2.0f );
	float v2 = pow( sin( H.GetPhi( ) ) / u[4], 2.0f );
	return u[2] * exp( -tan( H.GetTheta( ) ) * tan( H.GetTheta( ) ) * ( v1 + v2 ) );

}
UP partialDifferentialAshikhminModel(UP u, Vec3 tangent, Vec3 L){

	UP result(Nu,1);
	UP _u(u);
	for(size_t s=0; s<u.size(); s++){
		_u=u;
		_u[s]+=du[s];
		// u
		Vec3 n = Vec3( u[0], u[1] );
		Vec3 t = tangent;
		Vec3 b = Cross( t, n );
		Mat3x3 M = Mat3x3( t, b, n );
		Vec3 Vn = mul( M, N );
		Vec3 Ln = mul( M, L );
		Vec3 Hn = Normalize( Ln + Vn );
		// _u
		n = Vec3( _u[0], _u[1] );
		b = Normalize( Cross( tangent, n ) );
		t = Cross( b, n );
		M = Mat3x3( t, b, n );
		Vn = mul( M, N );
		Ln = mul( M, L );
		Vec3 _Hn = Normalize( Ln + Vn );
		//
		result[s] = ( AshikhminModel(u,Hn) - AshikhminModel(_u,_Hn) ) / du[s];
	}
	return result;
}

R8G8B8A8 vector2color(const Vec3 &normal){
	return R8G8B8A8( (tkd::BYTE)((normal.x+1.0f)/2.0f*255), (tkd::BYTE)((normal.y+1.0f)/2.0f*255), (tkd::BYTE)((normal.z+1.0f)/2.0f*255), 255);
}

int Phi(int p){
	if(p<0) return p+360;
	else if(p>=360) return p-360;
	else return p;
}
float Phi(float p){
	if(p<0.0f) return p+360.0f;
	else if(p>=360.0f) return p-360.0f;
	else return p;
}

Vec3is ThetaPhi(const Vec3is &d){
	if(d.theta>=0)
		return Vec3is(d.theta, Phi(d.phi));
	else
		return Vec3is(-d.theta, Phi(Phi(d.phi)-180));
}


float smoothing(int &theta, int &phi, const array2d<float> &refdata){
	
	float m = 0;
	int count = 0;
	for(int p=-1;p<2;p++){
		for(int t=-1;t<2;t++){
			if( noshadow((theta+t*5)/5,(phi+p*5)/5) ){
				m+=refdata.at((theta+t*5)/5,Phi(phi+p*5)/5);
				count++;
			}
		}
	}
	if(!count) return 0;

	return m/count;
}


//
// diffuse
//
float estimateLambertDiffuse(const array2d<float> &ref){

	int count=0;
	
	float error=0.0f;
	// initial error
	for(int phi=0;phi<360;phi+=5){
		for(int theta=0;theta<85;theta+=5){
			if(noshadow(theta/5,phi/5)){
				error += abs(ref.at(theta/5,phi/5));
				count++;
			}
		}
	}
	error/=count;

	// iterative estimation
	array2d< bool, array1d<bool> > specular(85/5,360/5,false);
	float average=0.0f;
	float current_error=error;
	do{
		// initialize
		error = current_error;
		current_error=0.0f;
		count=0;

		// average
		for(int phi=0;phi<360;phi+=5){
			for(int theta=0;theta<85;theta+=5){
				if(noshadow(theta/5,phi/5) && !specular.at(theta/5,phi/5)){
					average += ref.at(theta/5,phi/5);
					count++;
				}
			}
		}
		if(count==0) return 0.0f;
		average /= (float)count;

		count=0;
		// remove specular & evaluation
		for(int phi=0;phi<360;phi+=5){
			for(int theta=0;theta<85;theta+=5){
				if(noshadow(theta/5,phi/5) && !specular.at(theta/5,phi/5)){
					// remove specular
					if(ref.at(theta/5,phi/5)>average)
						specular.at(theta/5,phi/5) = true;
					else {// evaluation
						current_error += abs(ref.at(theta/5,phi/5)-average);
						count++;
					}
				}
			}
		}
		if(count==0) return 0.0f;

	}while(current_error>error);

	return average;
}

RGB<float> estimateDiffuseReflectance(const array2d<float> &rR, const array2d<float> &rG, const array2d<float> &rB){
	return RGB<float>(estimateLambertDiffuse(rR), estimateLambertDiffuse(rG), estimateLambertDiffuse(rB));
}


//
// specular
//
float decideStableThreshold(const array2d<float> &ref){
	
	std::vector<float> data;
	for(size_t p = 0; p < ref.height(); p++){
		for(size_t t = 0; t < ref.width(); t++){
			if(noshadow(t,p) && ref( t, p ) > dark )
				data.push_back( ref( t, p ) );
		}
	}
	// exception
	if(data.size() == 0) return 0.0f; 

	std::sort(data.begin(),data.end(),std::greater<float>());
	
	threshold = data[data.size()*0.2f];
	if(data.size() < 5) threshold = data.back(); // exception

	return data[0]; // return Maximum
}
/*
Vec3is searchMaximumReflectanceDirection(const array2d<float> &ref){
	
	Vec3is mrd;

	float max=0.0f;
	for(int phi=0;phi<360;phi+=5){
		for(int theta=0;theta<85;theta+=5){
			if(max<ref.at(theta/5,phi/5)){
				max = ref.at(theta/5,phi/5);
				mrd.theta = theta;
				mrd.phi = phi;
			}
		}
	}

	return mrd;
}
*/

Vec3fs estimateSpecularDirection(const array2d<float> &ref){

	//
	Vec3 S;
	for(size_t p = 0; p < ref.height(); p++){
		for(size_t t = 0; t < ref.width(); t++){
			if(noshadow(t,p) && ref( t, p )>=threshold)
				S += Vec3( radians(t*5.0f), radians(p*5.0f) ) * ref( t, p );
		}
	}
	S = Normalize(S);

	return Vec3fs( S.GetTheta( ), S.GetPhi( ) );
}

/*
Vec3fs estimateNormalDirection(const Vec3is &mrd, const array2d<float> &ref){
	
	array2d< bool, array1d<bool> > done(85/5,360/5,false);
	std::vector<Vec3is> candidates;
	std::vector<Vec3is> candidate_stock;
	const float maximum = ref.at(mrd.theta/5, mrd.phi/5);

	// initialize candidate_stock
	candidates.push_back(mrd);
	done.at(mrd.theta/5, mrd.phi/5)=true;
	//
	Vec3is neibor_dir = ThetaPhi(Vec3is(mrd.theta-5, mrd.phi));
	if(canAcquireReflectance(neibor_dir)) candidates.push_back(neibor_dir);
	done.at(neibor_dir.theta/5, neibor_dir.phi/5)=true;
	//
	neibor_dir = ThetaPhi(Vec3is(mrd.theta, mrd.phi-5));
	if(canAcquireReflectance(neibor_dir)) candidates.push_back(neibor_dir);
	done.at(neibor_dir.theta/5, neibor_dir.phi/5)=true;
	//
	neibor_dir = ThetaPhi(Vec3is(mrd.theta, mrd.phi+5));
	if(canAcquireReflectance(neibor_dir)) candidates.push_back(neibor_dir);
	done.at(neibor_dir.theta/5, neibor_dir.phi/5)=true;
	//
	neibor_dir = ThetaPhi(Vec3is(mrd.theta+5, mrd.phi));
	if(canAcquireReflectance(neibor_dir)) candidates.push_back(neibor_dir);
	done.at(neibor_dir.theta/5, neibor_dir.phi/5)=true;

	candidate_stock = candidates;
	while(!candidate_stock.empty()){
		std::vector<Vec3is> cs;
		for(std::vector<Vec3is>::iterator it=candidate_stock.begin(); it!=candidate_stock.end(); it++){
			// 8 neibors
			for(int phi=it->phi-5; phi<=it->phi+5; phi+=5){
				for(int theta=it->theta-5; theta<=it->theta+5; theta+=5){
					Vec3is dir = ThetaPhi(Vec3is(theta, phi));
					if(!done.at(dir.theta/5, dir.phi/5) && canAcquireReflectance(dir) && ref.at(dir.theta/5, dir.phi/5) > maximum*0.1f){
							cs.push_back(dir);
					}
					done.at(dir.theta/5, dir.phi/5)=true;
				}
			}
		}
		candidate_stock.clear();
		if(!cs.empty()){
			float t=0.0f;
			float p=0.0f;
			float a=0.0f;
			for(std::vector<Vec3is>::iterator it=candidates.begin(); it!=candidates.end(); it++){
				t+=it->theta*ref.at(it->theta/5,it->phi/5);
				p+=it->phi*ref.at(it->theta/5,it->phi/5);
				a+=ref.at(it->theta/5,it->phi/5);
			}
			for(std::vector<Vec3is>::iterator it=cs.begin(); it!=cs.end(); it++){
				t+=it->theta*ref.at(it->theta/5,it->phi/5);
				p+=it->phi*ref.at(it->theta/5,it->phi/5);
				a+=ref.at(it->theta/5,it->phi/5);
			}
			if( abs(t/a-mrd.theta)<5.0f && abs(p/a-mrd.phi)<5.0f ){
				candidate_stock=cs;
				candidates.insert(candidates.end(), cs.begin(), cs.end());
			}
		}
	}

	//std::ofstream fout("plot\\can.txt");
	float t=0.0f;
	float p=0.0f;
	float a=0.0f;
	for(std::vector<Vec3is>::iterator it=candidates.begin(); it!=candidates.end(); it++){
		t+=it->theta*ref.at(it->theta/5,it->phi/5);
		p+=it->phi*ref.at(it->theta/5,it->phi/5);
		a+=ref.at(it->theta/5,it->phi/5);
		//fout << it->theta << " " << it->phi << " " << ref.at(it->theta/5,it->phi/5) << std::endl;
	//	fout << radians((float)it->phi) << " " << radians(90.0f-(float)it->theta) << " " << ref.at(it->theta/5,it->phi/5) << std::endl;
	}

	return Vec3fs(t/a,p/a);
}
*/

Vec3 estimateTangent(const array2d<float> &ref, const Vec3 &n ){

	array2d< double > data( 3, ref.size( ) );
	array1d< float > eVal;
	array2d< float > eVec;
	array2d< float > weight;
	for(size_t p = 0; p < ref.height(); p++){
		for(size_t t = 0; t < ref.width(); t++){
			// h_tb
			Vec3 h = Vec3( radians(t*5.0f/2.0f), radians(p*5.0f) );
			float length = ref(t,p)*Dot(h,n);
			Vec3 h_tb = h*ref(t,p)-n*length;
			//
			data( 0, t + p * ref.width( ) ) = (double)h_tb.x;
			data( 1, t + p * ref.width( ) ) = (double)h_tb.y;
			data( 2, t + p * ref.width( ) ) = (double)h_tb.z;
		}
	}
	pca_averaged( data, eVal, eVec, weight );

//	float lambda1 = eVal[0] / (eVal[0] + eVal[1] + eVal[2]);
//	float lambda2 = eVal[1] / (eVal[0] + eVal[1] + eVal[2]);
//	float lambda3 = eVal[2] / (eVal[0] + eVal[1] + eVal[2]);

//	std::cout << "e1(t) : " << eVec.at(0,0) << " " << eVec.at(0,1) << " " << eVec.at(0,2) << std::endl;
//	std::cout << "e2(b) : " << eVec.at(1,0) << " " << eVec.at(1,1) << " " << eVec.at(1,2) << std::endl;
//	std::cout << "e3(?) : " << eVec.at(2,0) << " " << eVec.at(2,1) << " " << eVec.at(2,2) << std::endl;
//	std::cout << "lambda: " << lambda1 << " " << lambda2 << " " << lambda3 << std::endl;

	Vec3 t(eVec.at(0,0), eVec.at(0,1), eVec.at(0,2));
//	Vec3 b(eVec.at(1,0), eVec.at(1,1), eVec.at(1,2));

//	std::cout << "n : " << n << std::endl;
//	std::cout << "t : " << t << std::endl;
//	std::cout << "b : " << b << std::endl;
//	std::cout << "t*n : " << degrees(acos(Dot(t,n))) << std::endl;
//	std::cout << "b*n : " << degrees(acos(Dot(b,n))) << std::endl;

	return t;
}

int tX,tY;
void estimateInitialSpecularSigma(const array2d<float> &ref, AMP &amp ){
	
	//
	char filename[128]={};
	std::ofstream fout("pca.txt");
	std::ofstream foutHn("hn.txt");

	/*
	// mean
	size_t count=0;
	float mean = 0.0f;
	for(size_t phi = 0; phi < ref.height(); phi++){
		for(size_t theta = 0; theta < ref.width(); theta++){
			if(ref(theta,phi) > threshold){ // log(0)=error
				mean += ref(theta,phi);
				count++;
			}
		}
	}
	if(count == 0 || count == 1){
		amp.sigma_x = amp.sigma_y = min_sigma;
		return;
	}
	else
		mean /= (float)count;

	std::cout << threshold << " " << mean << " " << count << std::endl;
	*/

	threshold = threshold>amp.rho_s*0.2f ?  threshold : amp.rho_s*0.2f;

	//
	std::vector<Vec3> datas;
	array1d< float > eVal;
	array2d< float > eVec;
	array1d< float > ave;
	array2d< float > weight;
	//
	Vec3 n(amp.n.theta, amp.n.phi);
	Vec3 b = Cross( amp.t, n );
	Mat3x3 M = Mat3x3( amp.t, b, n );
	Vec3 Vn = mul( M, N );
	for(size_t phi = 0; phi < ref.height(); phi++){
		for(size_t theta = 0; theta < ref.width(); theta++){
			Vec3 L = Vec3( radians(theta*5.0f), radians(phi*5.0f) );
			Vec3 Ln = mul( M, L );
			Vec3 Hn = Normalize( Ln + Vn );
			float o = pow( tan(Hn.GetTheta()), 2.0f );
			float p = pow( cos(Hn.GetPhi()), 2.0f );
			if(ref(theta,phi) >= threshold){ // log(0)=error
				datas.push_back( Vec3(o*p, o-o*p, -log( ref(theta,phi) ) ) );
				fout << o*p << " " << o-o*p << " " << -log( ref(theta,phi) ) << std::endl;
			}
			foutHn << degrees(Hn.GetTheta()) << " " << Phi(degrees(Hn.GetPhi())) << " " << ref(theta,phi) << std::endl;
		}
		fout << std::endl;
		foutHn << std::endl;
	}

//	std::cout << "threshold: " << threshold << " datas.size() " << datas.size() << std::endl;
	if(datas.size()==0 || datas.size()==1){ // exception
		amp.sigma_x =  min_sigma;
		amp.sigma_y =  min_sigma;
		return;
	}


	array2d< double > data(3,datas.size());
	for(std::vector<Vec3>::size_type i = 0; i < datas.size(); i++){
		data( 0, i ) = datas[i].x;
		data( 1, i ) = datas[i].y;
		data( 2, i ) = datas[i].z;
	}

	pca( data, eVal, eVec, ave, weight );

//	float lambda1 = eVal[0] / (eVal[0] + eVal[1] + eVal[2]);
//	float lambda2 = eVal[1] / (eVal[0] + eVal[1] + eVal[2]);
//	float lambda3 = eVal[2] / (eVal[0] + eVal[1] + eVal[2]);
//	std::cout << "lambda: " << lambda1 << " " << lambda2 << " " << lambda3 << std::endl;
//	std::cout << "e3    : " << eVec.at(2,0) << " " << eVec.at(2,1) << " " << eVec.at(2,2) << std::endl;

	if( eVec.at( 2, 2 ) != 0.0f ){ // exception
		double A = eVec.at( 2, 0 ) / -eVec.at( 2, 2 );
		double B = eVec.at( 2, 1 ) / -eVec.at( 2, 2 );
		if(A != 0.0f) // exception
			amp.sigma_x = sqrt( 1.0f / abs(A) );
		if(B != 0.0f) // exception
		amp.sigma_y = sqrt( 1.0f / abs(B) );
//		std::cout << "A, B : " << A << " " << B << std::endl;
	}

	// exception
	if(amp.sigma_x>1.0f) amp.sigma_x = 1.0f;
	if(amp.sigma_y>1.0f) amp.sigma_y = 1.0f;


}

AMP estimateAshikhminModelParameter(const array2d<float> &ref){
	
	AMP amp;

	//
	amp.rho_s = decideStableThreshold(ref);
//	std::cout << "threshold " <<  threshold << std::endl;
//	std::cout << "max " << amp.rho_s << std::endl;
	
	// serch maximum reflectance direction
//	Vec3is mrd = searchMaximumReflectanceDirection(ref);
//	amp.rho_s = ref.at( mrd.theta/5, mrd.phi/5 );
//	std::cout << mrd.theta << " " <<  mrd.phi << " " << amp.rho_s << std::endl;


	// estimate normal direction n
	Vec3fs Ls = estimateSpecularDirection(ref);
	amp.n.theta = Ls.theta/2.0f;//radians(mrd.theta/2.0f);//
	amp.n.phi = Ls.phi;
//	std::cout << degrees(amp.n.theta) << " " <<  Phi(degrees(amp.n.phi)) << " " << amp.rho_s << std::endl;
//	amp.n.theta = radians(mrd.theta/2.0f);//Ls.theta/2.0f;
//	amp.n.phi = radians((float)mrd.phi);//Ls.phi;


	// estimate tangent t
	amp.t = estimateTangent( ref, Vec3(amp.n.theta, amp.n.phi) );

	// estimate sigma_x, sigma_y
	estimateInitialSpecularSigma(ref, amp );
	
	/*
	//
	char filename[128]={};
	sprintf(filename,"plot\\Dh%d-%d.txt",tX,tY);
	std::ofstream fout(filename);
	Vec3 n(amp.n.theta, amp.n.phi);
	Vec3 b = Cross( amp.t, n );
	Mat3x3 M = Mat3x3( amp.t, b, n );
	Vec3 Vn = mul( M, N );
	for(size_t phi = 0; phi < ref.height(); phi++){
		for(size_t theta = 0; theta < ref.width(); theta++){
			Vec3 L = Vec3( radians(theta*5.0f), radians(phi*5.0f) );
			Vec3 Ln = mul( M, L );
			Vec3 Hn = Normalize( Ln + Vn );
			float v1 = pow( cos( Hn.GetPhi( ) ) / amp.sigma_x, 2.0f );
			float v2 = pow( sin( Hn.GetPhi( ) ) / amp.sigma_y, 2.0f );
			float D = amp.rho_s * exp( -tan( Hn.GetTheta( ) ) * tan( Hn.GetTheta( ) ) * ( v1 + v2 ) );
			fout << degrees(Hn.GetTheta()) << " " << Phi(degrees(Hn.GetPhi())) << " " << D << std::endl;
		}
		fout << std::endl;
	}
	*/

	return amp;
}

AMP LevenbergMarquardt(const array2d<float> &ref, const AMP &amp){

	// 1. c
	float c = 0.0001f;
	
	// 2. u
	UP u(Nu,1);
	u[0] = amp.n.theta;
	u[1] = amp.n.phi;
	u[2] = amp.rho_s;
	u[3] = amp.sigma_x;
	u[4] = amp.sigma_y;
	//std::cout << "u : " << degrees(u[0]) << " " << Phi(degrees(u[1])) << " " << u[2] << " " << u[3] << " " << u[4] << std::endl;
//	std::cout << amp << std::endl;

	// 3. J
	float J = 0.0f;
	Vec3 n = Vec3( u[0], u[1] );
	Vec3 t = amp.t;
	Vec3 b = Cross( t, n );
	Mat3x3 M = Mat3x3( t, b, n );
	Vec3 Vn = mul( M, N );
	for(size_t p = 0; p < ref.height(); p++){
		for(size_t t = 0; t < ref.width(); t++){
			Vec3 L = Vec3( radians(t*5.0f), radians(p*5.0f) );
			Vec3 Ln = mul( M, L );
			Vec3 Hn = Normalize( Ln + Vn );
			if(noshadow(t,p))
				J += pow( ref.at(t,p) - AshikhminModel(u, Hn), 2.0f ) / 2.0f;
		}
	}
//	std::cout << "J : " << J << std::endl;


	const float delta = (float)1e-009;

	UP delta_u(Nu,1);
	size_t count = 0;
	do{
		// 4. ÞuJ and Hu
		UP delta_u_J(Nu,1);
		for(size_t p = 0; p < ref.height(); p++)
			for(size_t t = 0; t < ref.width(); t++)
				delta_u_J += ref.at(t,p) * partialDifferentialAshikhminModel(u, amp.t, Vec3( radians(t*5.0f), radians(p*5.0f) ));

		MAT H_u(Nu,Nu);
		for(size_t p = 0; p < ref.height(); p++){
			for(size_t t = 0; t < ref.width(); t++){
				UP delta_u_Fla = partialDifferentialAshikhminModel(u, amp.t, Vec3( radians(t*5.0f), radians(p*5.0f) ));
				MAT _H_u(delta_u_Fla * delta_u_Fla.t());
				H_u+=_H_u;
			}
		}

		// H_u !=0
		float zero = 0.0f;
		for(int i = 0; i < 5; i++){
			for(int j = 0; j < 5; j++)
				zero += abs(H_u.at(i,j));
		}
		if(zero == 0.0f) break;

		//std::cout << "H_u" << std::endl;
		//for(int i = 0; i < 5; i++){
		//	for(int j = 0; j < 5; j++)
		//		std::cout << H_u.at(i,j) << " ";
		//	std::cout << std::endl;
		//}
		//std::cout << "delta_u_J" << std::endl;
		//for(int i = 0; i < 5; i++)
		//	std::cout << delta_u_J[i] << " ";
		//std::cout << std::endl;


		float J_dash = 0.0f;
		UP u_dash(Nu,1);
		Vec3 t_dash;
		do{
			// 5. ƒ¢u
			// (Hu+cD[Hu])ƒ¢u=-ÞuJ
			MAT m(H_u);
			float amax = 0.0f;
			for (size_t i = 0; i < Nu; i ++)
				amax = (amax > m(i,i) ? amax : m(i,i));
			for(size_t j = 0; j < Nu; j++)
				m(j,j) += amax*c;

			MAT invm;
			try{
				invm = mist::inverse(m);
				delta_u = -invm*delta_u_J;
			}
			catch (char *str){
				u_dash=u;
				break;
			}

//			std::cout << "ƒ¢u : " << delta_u[0] << " " << delta_u[1] << " " << delta_u[2] << " " << delta_u[3] << " " << delta_u[4] << std::endl;

			// exception
			if( isnan(delta_u[0]) || isnan(delta_u[1]) || isnan(delta_u[2]) || isnan(delta_u[3]) || isnan(delta_u[4]) ){
				u_dash=u;
				break;
			}

			// 6. u'
			u_dash = u + delta_u;
			// exception
			if(cos(u_dash[0])<0.0f || u_dash[2]<0.0f ){
				u_dash = u;
				break;
			}
//			std::cout << "u' : " << u_dash[0] << " " << u_dash[1] << " " << u_dash[2] << " " << u_dash[3] << " " << u_dash[4] << std::endl;

			// 7. J'
			J_dash = 0.0f;
			n = Vec3( u_dash[0], u_dash[1] );
			t_dash = estimateTangent( ref, n );
			b = Cross( t_dash, n );
			M = Mat3x3( t_dash, b, n );
			Vn = mul( M, N );
			for(size_t p = 0; p < ref.height(); p++){
				for(size_t t = 0; t < ref.width(); t++){
					Vec3 L = Vec3( radians(t*5.0f), radians(p*5.0f) );
					Vec3 Ln = mul( M, L );
					Vec3 Hn = Normalize( Ln + Vn );
					if(noshadow(t,p))
						J_dash += pow( ref.at(t,p) - AshikhminModel(u_dash, Hn), 2.0f ) / 2.0f;
				}
			}
//			std::cout << "J' : " << J_dash << std::endl;

			// 8. J'>J c=10c goto 5.
			if(J_dash>J) c *= 10.0f;
			if(count>100) break;
			count++;
		}while(J_dash>J);

		// 9. J'<=J c=c/10, J=J', u=u'
		if(J_dash<=J) {
			c /= 10.0f;
			J = J_dash;
			u = u_dash;
			t = t_dash;
		}

//		std::cout << "c : " << c << std::endl;
//		std::cout << "ƒ¢u : " << delta_u[0] << " " << delta_u[1] << " " << delta_u[2] << " " << delta_u[3] << " " << delta_u[4] << std::endl;
//		std::cout << "u' : " << u_dash[0] << " " << u_dash[1] << " " << u_dash[2] << " " << u_dash[3] << " " << u_dash[4] << std::endl;

		// 10.  ||ƒ¢u|| < ƒÂ end.
		//		||ƒ¢u|| >= ƒÂ goto 4.
//		std::cout << "||ƒ¢u|| : " << norm(delta_u) << std::endl;
		if(norm(delta_u)<delta || count>100 || c<1e-045){
			if(count>100) std::cout << "over count" << std::endl;
			break;
		}
	}while(1);


	AMP _amp;
	_amp.n.theta	= u[0];
	_amp.n.phi		= u[1];
	_amp.t			= t;
	_amp.rho_s		= u[2];
	_amp.sigma_x	= u[3];
	_amp.sigma_y	= u[4];

	return _amp;
}


int main(){

	// make log file
	std::string logfile = inputpath + "\\" + "log.txt";
	std::ofstream flog(logfile.c_str());


	array2d< tkd::RGB<float> > reflectance_warp(T,P);
	array2d< tkd::RGB<float> > reflectance_weft(T,P);
	array2d< tkd::RGB<float> > irradiance_warp(T,P);
	array2d< tkd::RGB<float> > irradiance_weft(T,P);

	openEXR<float,2> HDRIwarp;
	openEXR<float,2> HDRIweft;
	bmp noshadowmap;
	if(HDRIwarp.load("warp.exr")){
		HDRIweft.load("weft.exr");
		noshadowmap.load("noshadow.bmp");
		for(size_t p = 0; p < P; p++){
			for(size_t t = 0; t < T; t++){
				//
				reflectance_warp(t,p) = HDRIwarp(t,p);
				reflectance_weft(t,p) = HDRIweft(t,p);
				noshadow(t,p) = noshadowmap.R(t,p);
			}
		}
	}
	else{

		// extract color
		char filename[128]={};
		for(size_t p = 0; p < P; p++){
			for(size_t t = 0; t < T; t++){

				int a = (int)(P*3/4)-(int)p;
				size_t _p = a < 0 ? P+a : a;
				std::cout << p << " " << _p << std::endl;
				flog << p << " " << _p << std::endl;


				// fabric
				sprintf( filename, "%03d%03d000000.exr", _p, maxnumT-t );
				std::string HDRIname = inputpath + "\\" + materialprefix + filename;
				std::cout << p*T+t << "/" << T*P << ": " << HDRIname.c_str() << std::endl;
				flog << p*T+t << "/" << T*P << ": " << HDRIname.c_str() << std::endl;
				//
				openEXR<float,2> HDRI;
				if(HDRI.load(HDRIname)){
					// warp
					tkd::RGB<float> c;
					for(size_t y = ys_warp; y < ye_warp; y++)
						for(size_t x = xs_warp; x < xe_warp; x++)
							c += HDRI(x,y);
					reflectance_warp(t,p)	=	c/(xe_warp-xs_warp)/(ye_warp-ys_warp);
					// weft
					c = tkd::RGB<float>(0.0f,0.0f,0.0f);
					for(size_t y = ys_weft; y < ye_weft; y++)
						for(size_t x = xs_weft; x < xe_weft; x++)
							c += HDRI(x,y);
					reflectance_weft(t,p)	=	c/(xe_weft-xs_weft)/(ye_weft-ys_weft);

					// white
					sprintf( filename, "%03d%03d000000.exr", _p, maxnumT-t );
					HDRIname = inputpath + "\\" + whiteprefix + filename;
					//
					HDRI.load(HDRIname);
					openEXR<float,2> HDRIwhite(HDRIname);
					// warp
					c = tkd::RGB<float>(0.0f,0.0f,0.0f);
					for(size_t y = ys_warp; y < ye_warp; y++)
						for(size_t x = xs_warp; x < xe_warp; x++)
							c += HDRI(x,y);
					irradiance_warp(t,p)	=	c/(xe_warp-xs_warp)/(ye_warp-ys_warp);
					// weft
					c = tkd::RGB<float>(0.0f,0.0f,0.0f);
					for(size_t y = ys_weft; y < ye_weft; y++)
						for(size_t x = xs_weft; x < xe_weft; x++)
							c += HDRI(x,y);
					irradiance_weft(t,p)	=	c/(xe_weft-xs_weft)/(ye_weft-ys_weft);

					//
					reflectance_warp(t,p) /= irradiance_warp(t,p);
					reflectance_weft(t,p) /= irradiance_weft(t,p);
					
					//
					noshadow(t,p)=1;
				}
				else {
					reflectance_warp(t,p) = tkd::RGB<float>(0.0f,0.0f,0.0f);
					reflectance_weft(t,p) = tkd::RGB<float>(0.0f,0.0f,0.0f);
					noshadow(t,p)=0;
				}

			}
		}

		//
		HDRIwarp.resize(T,P);
		HDRIweft.resize(T,P);
		noshadowmap.resize(T,P);
		for(size_t p = 0; p < P; p++){
			for(size_t t = 0; t < T; t++){
				HDRIwarp(t,p) = reflectance_warp(t,p);
				HDRIweft(t,p) = reflectance_weft(t,p);
				tkd::BYTE c = noshadow(t,p);
				noshadowmap(t,p) = tkd::R8G8B8A8(c,c,c,255);
			}
		}
		HDRIwarp.save("warp.exr");
		HDRIweft.save("weft.exr");
		noshadowmap.save("noshadow.bmp");
	}


	//
	//
	//
	bool warpdir = true;

	std::ofstream foutref("ref.txt");
	for(size_t p = 0; p < P; p++){
		for(size_t t = 0; t < T; t++){
			if(warpdir){
				refR.at(t,p) = reflectance_warp(t,p).r;
				refG.at(t,p) = reflectance_warp(t,p).g;
				refB.at(t,p) = reflectance_warp(t,p).b;
			}else{
				refR.at(t,p) = reflectance_weft(t,p).r;
				refG.at(t,p) = reflectance_weft(t,p).g;
				refB.at(t,p) = reflectance_weft(t,p).b;
			}
			foutref << t*5 << " " << p*5 << " " << refR.at(t,p) << std::endl;
		}
	}







	//
	// 01. separate reflectance into diffuse and specular reflectance
	//

	// estimate diffuse reflectance
	RGB<float> diffuse = estimateDiffuseReflectance(refR,refG,refB);
	std::cout << "diffuse : " << diffuse << std::endl;


	std::ofstream foutD("d.txt");
	// extract specular reflectance (reflectance minus diffuse reflectance)
	// and store microfacet distribution D (specular reflectance = D*F*G/cos(theta_i)/cos(theta_r) )
	Vec3f N(0.0f, 0.0f, 1.0f);
	Vec3f V(N);
	for(size_t p = 0; p < P; p++){
		for(size_t t = 0; t < T; t++){

			Vec3f L(radians(t*5.0f), radians(p*5.0f));
			Vec3f H = Normalize(L+V);

			// G
			float aa = (2.0f * Dot(N,H) * Dot(N,V) ) / Dot(V,H);
			float bb = (2.0f * Dot(N,H) * Dot(N,L) ) / Dot(V,H);
			float cc = 1.0f;
			float G = 0.0f;
			G = min( aa, bb );
			G = min( G, cc );
			float theta_p = 0.0f;
			if(warpdir){
				Vec3f Lp = Normalize(Vec3f(L.x,0.0f,L.z));
				theta_p = Lp.GetTheta();
			}
			else{
				Vec3f Lp = Normalize(Vec3f(0.0f,L.y,L.z));
				theta_p = Lp.GetTheta();
			}
			float t0 = acos(2.0f*cos(theta_p)-1.0f)-theta_p;
			G = (theta_p+PAI/2.0f-t0)/PAI;

			// F
			float c = Dot(L,H);
			float g = sqrt(n*n + c*c - 1);
			float T1 = ( (g-c)*(g-c)/(g+c)*(g+c) );
			float T2 = 1 + ( (c*(g+c)-1)*(c*(g+c)-1) )/( (c*(g-c)+1)*(c*(g-c)+1) );
			double F = T1*T2/2.0f;
			srefR.at(t,p) = max(0.0f,refR.at(t,p)-diffuse.r)*cos(radians(t*5.0f))/G/F;
			srefG.at(t,p) = max(0.0f,refG.at(t,p)-diffuse.g)*cos(radians(t*5.0f))/G/F;
			srefB.at(t,p) = max(0.0f,refB.at(t,p)-diffuse.b)*cos(radians(t*5.0f))/G/F;
			srefY(t,p) = srefR(t,p)*0.299f+srefG(t,p)*0.587f+srefB(t,p)*0.114f;
			foutD << t*5 << " " << p*5 << " " << srefR.at(t,p) << std::endl;
		}
	}

	//
	// 02. estimate Ashikhmin model parameter
	//

	// initialize parameter
	AMP ampR = estimateAshikhminModelParameter(srefR);
	AMP ampG = estimateAshikhminModelParameter(srefG);
	AMP ampB = estimateAshikhminModelParameter(srefB);
	AMP ampY = estimateAshikhminModelParameter(srefY);
	std::cout << ampR << std::endl;
	std::cout << ampG << std::endl;
	std::cout << ampB << std::endl;
	std::cout << ampY << std::endl;
	flog << ampR << std::endl;
	flog << ampG << std::endl;
	flog << ampB << std::endl;
	flog << ampY << std::endl;

	//
	std::ofstream foutsimD("simd.txt");
	for(size_t p = 0; p < P; p++){
		for(size_t t = 0; t < T; t++){
			Vec3f L(radians(t*5.0f), radians(p*5.0f));
			Vec3f n = Vec3( ampR.n.theta, ampR.n.phi );
			Vec3f ta = ampR.t;
			Vec3 b = Cross( ta, n );
			Mat3x3 M = Mat3x3( ta, b, n );
			Vec3 Vn = mul( M, N );
			Vec3 Ln = mul( M, L );
			Vec3 Hn = Normalize( Ln + Vn );
			UP u(Nu,1);
			u[0] = ampR.n.theta;
			u[1] = ampR.n.phi;
			u[2] = ampR.rho_s;
			u[3] = ampR.sigma_x;
			u[4] = ampR.sigma_y;
			foutsimD << t*5 << " " << p*5 << " " << AshikhminModel(u,Hn) << std::endl;
		}
	}

	// non-linear 
	du[2] = ampR.rho_s*0.1f; du[3] = ampR.sigma_x*0.1f; du[4] = ampR.sigma_y*0.1f;
	ampR = LevenbergMarquardt(srefR, ampR);
	du[2] = ampG.rho_s*0.1f; du[3] = ampG.sigma_x*0.1f; du[4] = ampG.sigma_y*0.1f;
	ampG = LevenbergMarquardt(srefG, ampG);
	du[2] = ampB.rho_s*0.1f; du[3] = ampB.sigma_x*0.1f; du[4] = ampB.sigma_y*0.1f;
	ampB = LevenbergMarquardt(srefB, ampB);
	du[2] = ampY.rho_s*0.1f; du[3] = ampY.sigma_x*0.1f; du[4] = ampY.sigma_y*0.1f;
	ampY = LevenbergMarquardt(srefY, ampY);
	std::cout << ampR << std::endl;
	std::cout << ampG << std::endl;
	std::cout << ampB << std::endl;
	std::cout << ampY << std::endl;
	flog << ampR << std::endl;
	flog << ampG << std::endl;
	flog << ampB << std::endl;
	flog << ampY << std::endl;
	
	//
	std::ofstream foutsimD2("simd2.txt");
	for(size_t p = 0; p < P; p++){
		for(size_t t = 0; t < T; t++){
			Vec3f L(radians(t*5.0f), radians(p*5.0f));
			Vec3f n = Vec3( ampR.n.theta, ampR.n.phi );
			Vec3f ta = ampR.t;
			Vec3 b = Cross( ta, n );
			Mat3x3 M = Mat3x3( ta, b, n );
			Vec3 Vn = mul( M, N );
			Vec3 Ln = mul( M, L );
			Vec3 Hn = Normalize( Ln + Vn );
			UP u(Nu,1);
			u[0] = ampR.n.theta;
			u[1] = ampR.n.phi;
			u[2] = ampR.rho_s;
			u[3] = ampR.sigma_x;
			u[4] = ampR.sigma_y;
			foutsimD2 << t*5 << " " << p*5 << " " << AshikhminModel(u,Hn) << std::endl;
		}
	}

	{
		// save BTF parameter map
		Vec3 n = Normalize( Vec3( ampR.n.theta, ampR.n.phi ) + Vec3( ampG.n.theta, ampG.n.phi ) + Vec3( ampB.n.theta, ampB.n.phi ) );//+ Vec3( (ampR.n.theta + ampG.n.theta + ampB.n.theta)/3.0f, (ampR.n.phi + ampG.n.phi + ampB.n.phi)/3.0f );
		Vec3 nR( ampR.n.theta, ampR.n.phi );//+ Vec3( (ampR.n.theta + ampG.n.theta + ampB.n.theta)/3.0f, (ampR.n.phi + ampG.n.phi + ampB.n.phi)/3.0f );
		Vec3 nG( ampG.n.theta, ampG.n.phi );
		Vec3 nB( ampB.n.theta, ampB.n.phi );
		Vec3 t = (ampR.t + ampG.t + ampB.t)/3.0f;
		RGB<float> specular( ampR.rho_s, ampG.rho_s, ampB.rho_s ); 
		RGB<float> SigmaX( ampR.sigma_x, ampG.sigma_x, ampB.sigma_x ); 
		RGB<float> SigmaY( ampR.sigma_y, ampG.sigma_y, ampB.sigma_y );
	}
			
	return 0;
}