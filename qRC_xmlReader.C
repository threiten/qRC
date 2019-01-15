#include <iostream>
using namespace std;


struct qRC_Input{
  float qRC_Input_pt_;
  float qRC_Input_ScEta_;
  float qRC_Input_Phi_;
  float qRC_Input_rho_;
  float qRC_Input_covaIEtaIPhi_;
  float qRC_Input_S4_;
  float qRC_Input_R9_;
  float qRC_Input_phiWidth_;
  float qRC_Input_sigmaIEtaIEta_;
  float qRC_Input_etaWidth_;
};



TMVA::Reader* bookReaders(const string &xmlfilename, qRC_Input &inp){

  string mvamethod = "BDTG";
  
  TMVA::Reader* qRCReader_ = new TMVA::Reader( "!Color:Silent" );
  
  qRCReader_->AddVariable( "f0", &inp.qRC_Input_pt_ );
  qRCReader_->AddVariable( "f1", &inp.qRC_Input_ScEta_ );
  qRCReader_->AddVariable( "f2", &inp.qRC_Input_Phi_ );
  qRCReader_->AddVariable( "f3", &inp.qRC_Input_rho_ );
  qRCReader_->AddVariable( "f4", &inp.qRC_Input_covaIEtaIPhi_ );
  qRCReader_->AddVariable( "f5", &inp.qRC_Input_S4_);
  qRCReader_->AddVariable( "f6", &inp.qRC_Input_R9_ );
  qRCReader_->AddVariable( "f7", &inp.qRC_Input_phiWidth_ );
  qRCReader_->AddVariable( "f8", &inp.qRC_Input_sigmaIEtaIEta_ );
  qRCReader_->AddVariable( "f9", &inp.qRC_Input_etaWidth_ );

  qRCReader_->BookMVA( mvamethod.c_str(), xmlfilename );

  return qRCReader_;
}
