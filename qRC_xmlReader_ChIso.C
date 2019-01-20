#include <iostream>
using namespace std;

class qRC_Input_ChIso
{
public:
  qRC_Input_ChIso()
  {
    qRC_Input_pt_=0;
    qRC_Input_ScEta_=0;
    qRC_Input_Phi_=0;
    qRC_Input_rho_=0;
    qRC_Input_chIso03_=0;
    qRC_Input_chIso03worst_=0;
    qRC_Input_rand01_=0;
  }

  float qRC_Input_pt_;
  float qRC_Input_ScEta_;
  float qRC_Input_Phi_;
  float qRC_Input_rho_;
  float qRC_Input_chIso03_;
  float qRC_Input_chIso03worst_;
  float qRC_Input_rand01_;
};

TMVA::Reader* bookReaderFinalReg(const string &xmlfilename, qRC_Input_ChIso &inp){

  string mvaNameFinalReg = "finalReg";
  
  TMVA::Reader* qRCReader_finalReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_finalReg->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_finalReg->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_finalReg->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_finalReg->AddVariable( "f3", &inp.qRC_Input_rho_ );
  qRCReader_finalReg->AddVariable( "f4", &inp.qRC_Input_chIso03_ );
  qRCReader_finalReg->AddVariable( "f5", &inp.qRC_Input_chIso03worst_ );

  qRCReader_finalReg->BookMVA( mvaNameFinalReg.c_str(), xmlfilename );

  return qRCReader_finalReg;
}

TMVA::Reader* bookReaderTailRegChIso(const string &var, const string &xmlfilename, qRC_Input_ChIso &inp){

  string mvaNameTailReg = "tailReg";
  
  TMVA::Reader* qRCReader_tailReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_tailReg->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_tailReg->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_tailReg->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_tailReg->AddVariable( "f3", &inp.qRC_Input_rho_ );
  if (var=="ChIso03"){
    qRCReader_tailReg->AddVariable( "f4", &inp.qRC_Input_chIso03worst_ );}
  else if (var=="ChIso03worst"){    
    qRCReader_tailReg->AddVariable( "f4", &inp.qRC_Input_chIso03_ );}
  qRCReader_tailReg->AddVariable( "f5", &inp.qRC_Input_rand01_ );

  qRCReader_tailReg->BookMVA( mvaNameTailReg.c_str(), xmlfilename );

  return qRCReader_tailReg;
}

TMVA::Reader* bookReader3CatClf(const string &xmlfilename, qRC_Input_ChIso &inp){

  string mvaNameClf = "3CatClf";
  
  TMVA::Reader* qRCReader_3CatClf = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_3CatClf->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_3CatClf->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_3CatClf->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_3CatClf->AddVariable( "f3", &inp.qRC_Input_rho_ );

  qRCReader_3CatClf->BookMVA( mvaNameClf.c_str(), xmlfilename );

  return qRCReader_3CatClf;
}
