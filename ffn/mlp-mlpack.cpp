// g++ mlp-mlpack.cpp -o mlp-mlpack -std=c++11 -larmadillo -lmlpack  

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>


using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;
using namespace arma;
using namespace std;


int main(){
	
	mat data;

	data::Load("foo.csv",data,true);

	mat traindata = data.submat(0,0,data.n_rows-2,data.n_cols-6);
	mat trainlabels = data.submat(data.n_rows-1,0,data.n_rows-1,data.n_cols-6);


	mat testdata = data.submat(0,data.n_cols-5,data.n_rows-2,data.n_cols-1);
	mat testlabels = data.submat(data.n_rows-1,data.n_cols-5,data.n_rows-1,data.n_cols-1);

	// cout << testdata <<endl;
	// cout << testlabels <<endl;

	FFN<MeanSquaredError<>, RandomInitialization> model;

	model.Add<Linear<> >(traindata.n_rows,8);
	model.Add<SigmoidLayer<> >();
	model.Add<Linear<> >(8,8);
	model.Add<SigmoidLayer<> >();
	model.Add<Linear<> >(8,1);
	model.Add<SigmoidLayer<> >();
	

	for (int i = 0; i < 4; ++i)
	{
		model.Train(traindata, trainlabels);
	}
	

	mat assignments;

	model.Predict(testdata, assignments);

	cout<<"Predictions    : "<<assignments<<endl;
	cout<<"Correct Labels : "<<testlabels<<endl;

	// data::Save("cov.csv",cov,true);
	return 0;
}