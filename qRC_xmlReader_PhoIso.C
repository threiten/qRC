#include <iostream>
using namespace std;

class qRC_Input_Iso
{
public:
  qRC_Input_Iso()
  {
    qRC_Input_pt_=0;
    qRC_Input_ScEta_=0;
    qRC_Input_Phi_=0;
    qRC_Input_rho_=0;
    qRC_Input_phoIso_=0;
    qRC_Input_rand01_=0;
  }

  float qRC_Input_pt_;
  float qRC_Input_ScEta_;
  float qRC_Input_Phi_;
  float qRC_Input_rho_;
  float qRC_Input_phoIso_;
  float qRC_Input_rand01_;
};

TMVA::Reader* bookReaderFinalReg(const string &xmlfilename, qRC_Input_Iso &inp){

  string mvaNameFinalReg = "finalReg";
  
  TMVA::Reader* qRCReader_finalReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_finalReg->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_finalReg->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_finalReg->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_finalReg->AddVariable( "f3", &inp.qRC_Input_rho_ );
  qRCReader_finalReg->AddVariable( "f4", &inp.qRC_Input_phoIso_ );

  qRCReader_finalReg->BookMVA( mvaNameFinalReg.c_str(), xmlfilename );

  return qRCReader_finalReg;
}

TMVA::Reader* bookReaderTailReg(const string &xmlfilename, qRC_Input_Iso &inp){

  string mvaNameTailReg = "tailReg";
  
  TMVA::Reader* qRCReader_tailReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_tailReg->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_tailReg->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_tailReg->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_tailReg->AddVariable( "f3", &inp.qRC_Input_rho_ );
  qRCReader_tailReg->AddVariable( "f4", &inp.qRC_Input_rand01_ );

  qRCReader_tailReg->BookMVA( mvaNameTailReg.c_str(), xmlfilename );

  return qRCReader_tailReg;
}

TMVA::Reader* bookReaderpotClf(const string &xmlfilename, qRC_Input_Iso &inp){

  string mvaNameClf = "potClf";
  
  TMVA::Reader* qRCReader_potClf = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_potClf->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_potClf->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_potClf->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_potClf->AddVariable( "f3", &inp.qRC_Input_rho_ );

  qRCReader_potClf->BookMVA( mvaNameClf.c_str(), xmlfilename );

  return qRCReader_potClf;
}
