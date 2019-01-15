#include <iostream>
using namespace std;

class qRC_Input
{
public:
  qRC_Input()
  {
    qRC_Input_pt_=0;
    qRC_Input_ScEta_=0;
    qRC_Input_Phi_=0;
    qRC_Input_rho_=0;
    qRC_Input_phoIso_=0;
    qRC_Input_rand01_=0
  }

  float qRC_Input_pt_;
  float qRC_Input_ScEta_;
  float qRC_Input_Phi_;
  float qRC_Input_rho_;
  float qRC_Input_phoIso_;
  float qRC_Input_rand01_;
};

TMVA::Reader* bookReaderFinalReg(const string &xmlfilename, qRC_Input &inp){

  //string mvamethod = "BDT";
  
  TMVA::Reader* qRCReader_finalReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_finalReg->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_finalReg->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_finalReg->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_finalReg->AddVariable( "f3", &inp.qRC_Input_rho_ );
  qRCReader_finalReg->AddVariable( "f4", &inp.qRC_Input_phoIso_ );

  qRCReader_finalReg->BookMVA( "finalReg", xmlfilename );

  return qRCReader_finalReg;
}

TMVA::Reader* bookReaderTailReg(const string &xmlfilename, qRC_Input &inp){

  //string mvamethod = "BDT";
  
  TMVA::Reader* qRCReader_tailReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_tailReg->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_tailReg->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_tailReg->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_tailReg->AddVariable( "f3", &inp.qRC_Input_rho_ );
  qRCReader_tailReg->AddVariable( "f4", &inp.qRC_Input_rand01_ );

  qRCReader_tailReg->BookMVA( "tailReg", xmlfilename );

  return qRCReader_tailReg;
}

TMVA::Reader* bookReaderDataClf(const string &xmlfilename, qRC_Input &inp){

  //string mvamethod = "BDT";
  
  TMVA::Reader* qRCReader_tailReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_dataClf->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_dataClf->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_dataClf->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_dataClf->AddVariable( "f3", &inp.qRC_Input_rho_ );

  qRCReader_dataClf->BookMVA( "dataClf", xmlfilename );

  return qRCReader_dataClf;
}

TMVA::Reader* bookReaderMcClf(const string &xmlfilename, qRC_Input &inp){

  //string mvamethod = "BDT";
  
  TMVA::Reader* qRCReader_tailReg = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_mcClf->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_mcClf->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_mcClf->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_mcClf->AddVariable( "f3", &inp.qRC_Input_rho_ );

  qRCReader_mcClf->BookMVA( "mcClf", xmlfilename );

  return qRCReader_mcClf;
}
