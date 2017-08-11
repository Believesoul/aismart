#ifndef GUI_DIALOGS_HOME_HPP_INCLUDED
#define GUI_DIALOGS_HOME_HPP_INCLUDED

#include "gui/dialogs/dialog.hpp"
#include "rtc_client.hpp"

#include <opencv2/core.hpp>
#include <tensorflow/core/public/session.h>

class display;

namespace gui2 {

class ttoggle_button;
class ttrack;
class tslider;

class thome: public tdialog, public tworker
{
public:
	enum tresult {CHAT = 1};

	enum {mouse, inception5h, classifier, detector, pr, chat};

	explicit thome(display& disp);
	~thome();

private:
	/** Inherited from tdialog. */
	void pre_show(CVideo& video, twindow& window);

	/** Inherited from tdialog. */
	void post_show(twindow& window);

	/** Inherited from tdialog, implemented by REGISTER_DIALOG. */
	virtual const std::string& window_id() const;

	void DoWork() override;
	void OnWorkStart() override;
	void OnWorkDone() override;

	void example_mouse();
	void example_inception5h();
	void example_classifier();
	void example_detector();
	void example_pr();
	void did_example_item_changed(ttoggle_button& widget);
	void did_draw_paper(ttrack& widget, const SDL_Rect& draw_rect, const bool bg_drawn);
	void did_left_button_down_paper(ttrack& widget, const tpoint& coordinate);
	void did_mouse_leave_paper(ttrack& widget, const tpoint& first, const tpoint& last);
	void did_mouse_motion_paper(ttrack& widget, const tpoint& first, const tpoint& last);
	void did_slider_value_changed(tslider& widget, int value, int at);

	void did_first_drawn(twindow& window) override;

	std::string example_inception5h_internal();
	std::string example_inception5h_internal2(surface& surf);
	std::string example_detector_internal(surface& surf);
	std::string example_pr_internal(surface& surf);

	SDL_Rect app_did_draw_frame(bool remote, cv::Mat& frame, const SDL_Rect& draw_rect);

	void stop_avcapture();
	void avcapture_switch_scenario(const std::string& name);

private:
	display & disp_;
	ttrack* paper_;
	std::vector<image::tblit> blits_;

	std::pair<std::string, std::unique_ptr<tensorflow::Session> > current_session_;
	std::pair<bool, surface> current_surf_; // first: valid. now can recognition.

	threading::mutex recognition_mutex_;
	threading::mutex variable_mutex_;
	bool recognition_thread_running_;

	std::string result_;
	std::vector<std::pair<float, SDL_Rect> > rects_;
	std::vector<std::string> label_strings_;
	std::vector<float> locations_;

	class tsetting_lock
	{
	public:
		tsetting_lock(thome& home)
			: home_(home)
		{
			VALIDATE(!home_.setting_, null_str);
			home_.setting_ = true;
		}
		~tsetting_lock()
		{
			home_.setting_ = false;
		}

	private:
		thome& home_;
	};
	bool setting_;

	std::unique_ptr<tavcapture> avcapture_;
	surface persist_surf_;
	surface temperate_surf_;

	int current_example_;
	tpoint last_coordinate_;
	cv::RNG rng_;
};

} // namespace gui2

#endif

